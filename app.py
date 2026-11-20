"""Streamlit interface for the RAG Knowledge Assistant."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

from chunking import chunk_pages
from embeddings import embed_documents, embed_query
from generation import (
    FALLBACK_ANSWER,
    FALLBACK_GENERATION_MODEL,
    GENERATION_MODEL,
    generate_grounded_answer,
    validate_citations,
)
from ingestion import IMAGE_EXTENSIONS, extract_document
from retrieval import rank_chunks


MAX_UPLOAD_MB = 10
OCR_PACKAGES = ("pymupdf", "pytesseract", "cv2")
# Stable, beginner-friendly defaults. Keep these fixed in the UI so the app is
# ready to use; change them here when intentionally running an experiment.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
MINIMUM_SCORE = 0.35


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, .18);}
    .app-subtitle {color: #9aa4b2; margin-top: -.5rem; margin-bottom: 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _api_key() -> str:
    """Prefer project secrets so stale machine environment variables cannot win."""
    try:
        secret_google_key = str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
        secret_gemini_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        secret_google_key = ""
        secret_gemini_key = ""

    return (
        secret_gemini_key
        or secret_google_key
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


def _setting(name: str, default: str) -> str:
    """Read an optional model setting from Streamlit secrets or the environment."""
    try:
        secret_value = str(st.secrets.get(name, "")).strip()
    except Exception:
        secret_value = ""
    return secret_value or os.environ.get(name, "").strip() or default


@st.cache_resource(show_spinner=False)
def _genai_client(api_key: str):
    """Reuse a Gemini client across Streamlit reruns to avoid repeated setup logs."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_resource(show_spinner="Loading the document layout model...")
def _load_layout_model(weights_path: str):
    from ocr_layout import load_model

    return load_model(Path(weights_path))


def _initialize_state() -> None:
    st.session_state.setdefault("parsed_documents", {})
    st.session_state.setdefault("ocr_results", {})
    st.session_state.setdefault("ocr_auto_modes", {})
    st.session_state.setdefault("manual_ocr_enabled", {})
    st.session_state.setdefault("knowledge_sources", {})
    st.session_state.setdefault("indexed_extraction_modes", {})
    st.session_state.setdefault("index_attempts", {})
    st.session_state.setdefault("index_errors", {})
    st.session_state.setdefault("chat_history", [])


def _documents_from_uploads(uploaded_files):
    """Return source name, bytes and stable content ID for selected documents."""
    documents = []
    for uploaded in uploaded_files or []:
        data = uploaded.getvalue()
        documents.append((uploaded.name, data, _digest(data)))
    return documents


def _pages_for_document(
    source_name: str,
    content: bytes,
    content_id: str,
    use_ocr: bool = False,
):
    """Extract once per session and overlay reviewed OCR text where available."""
    parsed = st.session_state["parsed_documents"]
    if content_id not in parsed:
        parsed[content_id] = extract_document(source_name, content)

    pages = deepcopy(parsed[content_id])
    manual_ocr_enabled = st.session_state["manual_ocr_enabled"].get(content_id, False)
    ocr_pages = (
        st.session_state["ocr_results"].get(content_id, {})
        if use_ocr or manual_ocr_enabled
        else {}
    )
    for page in pages:
        label = page.get("label", "")
        if label.startswith(("Page ", "Photo ")):
            page_number = int(label.split()[-1])
            if page_number in ocr_pages:
                page["text"] = ocr_pages[page_number]["text"]
                page["ocr_regions"] = ocr_pages[page_number]["regions"]
    return pages


def _show_document_preview(
    source_name: str,
    content: bytes,
    content_id: str,
    use_ocr: bool = False,
    ocr_method: str = "Layout model + Tesseract",
) -> list[dict]:
    """Preview extracted content and optionally apply OCR to PDF pages."""
    try:
        pages = _pages_for_document(source_name, content, content_id, use_ocr=use_ocr)
    except Exception as error:
        st.error(f"Could not read **{source_name}**: {error}")
        return []

    if Path(source_name).suffix.lower() == ".pdf":
        missing_packages = [
            package for package in OCR_PACKAGES
            if importlib.util.find_spec(package) is None
        ]
        from ocr_layout import DEFAULT_WEIGHTS

        if missing_packages:
            st.info(
                "OCR is optional and is not installed in this environment. Missing: "
                + ", ".join(missing_packages)
                + ". Install the packages in `requirements-ocr.txt` and install Tesseract."
            )
        else:
            from ocr_layout import DEFAULT_WEIGHTS, ocr_pdf_pages

            has_layout_model = (
                importlib.util.find_spec("ultralytics") is not None
                and DEFAULT_WEIGHTS.is_file()
            )
            page_numbers = [
                int(page["label"].split()[-1])
                for page in pages
                if page["label"].startswith("Page ")
            ]

            if use_ocr and page_numbers:
                previous_mode = st.session_state["ocr_auto_modes"].get(content_id)
                if previous_mode != ocr_method:
                    if ocr_method == "Layout model + Tesseract" and not has_layout_model:
                        st.caption(
                            "The layout model is unavailable, so this run will use Tesseract only. "
                            "The model files and OCR options have been kept."
                        )
                    try:
                        with st.spinner(f"Using OCR to read {source_name}..."):
                            model = (
                                _load_layout_model(str(DEFAULT_WEIGHTS))
                                if ocr_method == "Layout model + Tesseract" and has_layout_model
                                else None
                            )
                            results = ocr_pdf_pages(content, page_numbers, model)
                        st.session_state["ocr_results"][content_id] = results
                        st.session_state["ocr_auto_modes"][content_id] = ocr_method
                        pages = _pages_for_document(
                            source_name, content, content_id, use_ocr=True
                        )
                        st.success("OCR finished. Review the recognized text before indexing.")
                    except Exception as error:
                        st.error(f"OCR could not process this PDF: {error}")

            empty_pages = [page for page in pages if not page.get("text", "").strip()]
            pages_to_ocr = empty_pages or pages
            manual_page_numbers = [
                int(page["label"].split()[-1])
                for page in pages_to_ocr
                if page["label"].startswith("Page ")
            ]
            if empty_pages:
                button_label = f"Run OCR on {len(manual_page_numbers)} page(s) without selectable text"
                st.caption("OCR will target only pages where no text was extracted.")
            else:
                button_label = f"Run OCR manually on all {len(manual_page_numbers)} PDF page(s)"
                st.caption(
                    "Use this if extracted text looks incomplete or out of order. "
                    "Review OCR output before indexing."
                )
            if st.button(
                button_label,
                key=f"ocr_{content_id}",
                disabled=not manual_page_numbers,
            ):
                try:
                    with st.spinner("Recognizing text from PDF pages..."):
                        model = (
                            _load_layout_model(str(DEFAULT_WEIGHTS))
                            if ocr_method == "Layout model + Tesseract" and has_layout_model
                            else None
                        )
                        results = ocr_pdf_pages(content, manual_page_numbers, model)
                    st.session_state["ocr_results"][content_id] = results
                    st.session_state["manual_ocr_enabled"][content_id] = True
                    if use_ocr:
                        st.session_state["ocr_auto_modes"][content_id] = ocr_method
                    st.success("OCR finished. Review the recognized text and labels below.")
                    st.rerun()
                except Exception as error:
                    st.error(f"OCR could not process this PDF: {error}")

    elif Path(source_name).suffix.lower() in IMAGE_EXTENSIONS:
        missing_packages = [
            package
            for package in ("pytesseract", "cv2", "PIL")
            if importlib.util.find_spec(package) is None
        ]
        if missing_packages:
            st.info(
                "Photo OCR needs the optional packages in `requirements-ocr.txt` and a local Tesseract installation."
            )
        elif not use_ocr:
            st.info("Photos require OCR. Enable **Use OCR for PDFs and photos** above to index this image.")
        else:
            from ocr_layout import DEFAULT_WEIGHTS, ocr_image

            has_layout_model = (
                importlib.util.find_spec("ultralytics") is not None
                and DEFAULT_WEIGHTS.is_file()
            )
            previous_mode = st.session_state["ocr_auto_modes"].get(content_id)
            if previous_mode != ocr_method:
                try:
                    with st.spinner(f"Using OCR to read {source_name}..."):
                        model = (
                            _load_layout_model(str(DEFAULT_WEIGHTS))
                            if ocr_method == "Layout model + Tesseract" and has_layout_model
                            else None
                        )
                        st.session_state["ocr_results"][content_id] = {
                            1: ocr_image(content, model)
                        }
                    st.session_state["ocr_auto_modes"][content_id] = ocr_method
                    pages = _pages_for_document(
                        source_name, content, content_id, use_ocr=True
                    )
                    st.success("Photo OCR finished. Recognized text will be indexed.")
                except Exception as error:
                    st.error(f"OCR could not process this photo: {error}")

    empty_pages = [page for page in pages if not page.get("text", "").strip()]
    total_chars = sum(len(page.get("text", "")) for page in pages)
    if empty_pages:
        missing_labels = ", ".join(page["label"] for page in empty_pages)
        st.warning(f"{source_name}: no text was extracted from {missing_labels}.")
    else:
        st.caption(f"{source_name}: {len(pages)} page/section(s), {total_chars:,} characters")

    with st.expander(f"Preview extracted content: {source_name}"):
        if not pages:
            st.info("No text pages were extracted.")
        for page in pages:
            st.markdown(f"**{page['label']}**")
            if page.get("text", "").strip():
                st.text_area(
                    f"Extracted text — {page['label']}",
                    page["text"],
                    height=150,
                    key=f"preview_{content_id}_{page['label']}",
                    label_visibility="collapsed",
                )
            else:
                st.caption("No text available on this page.")
            if page.get("ocr_regions"):
                st.dataframe(
                    [
                        {
                            "Detected layout": region["layout_label"],
                            "Confidence": region["layout_confidence"],
                            "Recognized text": region["text"][:240],
                        }
                        for region in page["ocr_regions"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
    return pages


def _build_knowledge_base(documents, prepared_pages, chunk_size: int, overlap: int, api_key: str):
    client = _genai_client(api_key)
    current_sources = st.session_state["knowledge_sources"]
    added_count = 0
    errors = []
    pending_sources = []
    all_chunks = []

    with st.spinner("Preparing chunks and creating document embeddings..."):
        for source_name, _content, content_id in documents:
            if content_id in current_sources:
                continue
            pages = prepared_pages.get(content_id, [])
            chunks = chunk_pages(pages, source_name, chunk_size, overlap)
            if not chunks:
                errors.append(f"{source_name}: no text was available to index.")
                continue
            start = len(all_chunks)
            all_chunks.extend(chunks)
            pending_sources.append((source_name, content_id, pages, chunks, start))

        if all_chunks:
            all_vectors = embed_documents(
                client, [str(chunk["text"]) for chunk in all_chunks]
            )
            for source_name, content_id, pages, chunks, start in pending_sources:
                vectors = all_vectors[start : start + len(chunks)]
                if len(vectors) != len(chunks):
                    raise RuntimeError("Gemini returned an incomplete set of document embeddings.")
                current_sources[content_id] = {
                    "source": source_name,
                    "chunks": chunks,
                    "vectors": vectors,
                    "page_count": len(pages),
                }
                added_count += 1

    return added_count, errors


def _answer_question(
    question: str,
    minimum_score: float,
    top_k: int,
    api_key: str,
    progress=None,
) -> dict:
    from google import genai

    all_chunks = []
    all_vectors = []
    for source in st.session_state["knowledge_sources"].values():
        all_chunks.extend(source["chunks"])
        all_vectors.extend(source["vectors"])

    if not all_chunks:
        return {"answer": FALLBACK_ANSWER, "citations": [], "evidence": [], "status": "fallback"}

    client = _genai_client(api_key)
    if progress:
        progress("Creating a search vector for your question…")
    query_vector = embed_query(client, question)
    if progress:
        progress("Finding the most relevant document passages…")
    retrieved = rank_chunks(query_vector, all_chunks, all_vectors, top_k=top_k)
    eligible = [item for item in retrieved if item["score"] >= minimum_score]
    if not eligible:
        return {
            "answer": FALLBACK_ANSWER,
            "citations": [],
            "evidence": retrieved,
            "status": "fallback",
        }

    evidence = [
        {"id": f"S{index}", "chunk": item["chunk"]}
        for index, item in enumerate(eligible, start=1)
    ]
    if progress:
        progress("Writing an answer from the retrieved evidence…")
    raw_answer = generate_grounded_answer(
        client,
        question,
        evidence,
        model=_setting("GEMINI_MODEL", GENERATION_MODEL),
        fallback_model=_setting("GEMINI_FALLBACK_MODEL", FALLBACK_GENERATION_MODEL),
    )
    checked = validate_citations(raw_answer, {source["id"] for source in evidence})
    by_id = {source["id"]: source["chunk"] for source in evidence}
    scores_by_id = {
        f"S{index}": item["score"]
        for index, item in enumerate(eligible, start=1)
    }
    citations = [
        {
            "id": citation_id,
            "chunk": by_id[citation_id],
            "score": scores_by_id[citation_id],
        }
        for citation_id in checked["citations"]
    ]
    return {
        "answer": checked["answer"],
        "citations": citations,
        "evidence": eligible,
        "status": checked["status"],
    }


def _render_history() -> None:
    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                if turn.get("citations"):
                    st.markdown("**Sources**")
                    for citation in turn["citations"]:
                        chunk = citation["chunk"]
                        with st.expander(
                            f"{citation['id']} · {chunk['source']} · {chunk['page']} · relevance {citation['score']:.2f}"
                        ):
                            st.write(chunk["text"])
                if turn.get("evidence"):
                    with st.expander("Retrieved passages"):
                        for match in turn["evidence"]:
                            chunk = match["chunk"]
                            st.markdown(
                                f"**{chunk['source']} · {chunk['page']} · "
                                f"relevance {match['score']:.2f}**"
                            )
                            st.write(chunk["text"])


_initialize_state()
api_key = _api_key()

st.title("RAG Knowledge Assistant")
st.markdown(
    '<p class="app-subtitle">Ask questions about your approved documents. Answers are grounded in retrieved passages and show their sources.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Your documents")
    uploaded_files = st.file_uploader(
        "Add PDF, photo, TXT, or Markdown files",
        type=["pdf", "txt", "md", *sorted(suffix.lstrip(".") for suffix in IMAGE_EXTENSIONS)],
        accept_multiple_files=True,
        max_upload_size=MAX_UPLOAD_MB,
        help=f"Each file can be up to {MAX_UPLOAD_MB} MB.",
    )
    use_ocr = st.checkbox(
        "Use OCR for PDFs and photos",
        value=False,
        help=(
            "When enabled, OCR runs locally on each selected PDF page and photo. Photos are never indexed without OCR."
        ),
    )
    from ocr_layout import DEFAULT_WEIGHTS

    layout_model_available = (
        importlib.util.find_spec("ultralytics") is not None and DEFAULT_WEIGHTS.is_file()
    )
    ocr_method = st.selectbox(
        "OCR method",
        ["Layout model + Tesseract", "Tesseract only"],
        index=0 if layout_model_available else 1,
        disabled=not use_ocr,
        help=(
            "The layout model finds text regions; Tesseract recognizes the text. "
            "Tesseract-only mode keeps OCR available without loading the model."
        ),
    )
    selected_documents = _documents_from_uploads(uploaded_files)

    if not api_key:
        st.warning(
            "Add a valid `GEMINI_API_KEY` to `.streamlit/secrets.toml` to index documents "
            "and ask questions. See the setup instructions in the README."
        )
    else:
        st.caption("Documents are indexed automatically after extraction and OCR review.")

    sources = st.session_state["knowledge_sources"]
    st.divider()
    st.subheader(f"Indexed in this session · {len(sources)}")
    if sources:
        for content_id, source in list(sources.items()):
            col_name, col_remove = st.columns([4, 1])
            col_name.caption(f"{source['source']} · {len(source['chunks'])} passages")
            if col_remove.button("×", key=f"remove_{content_id}", help="Remove from this session"):
                del sources[content_id]
                st.rerun()
    if st.button("Clear chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

with st.expander("Privacy and data use"):
    st.markdown(
        "Documents are processed in this app session. When a document is indexed, "
        "document chunks are sent to Gemini to create embeddings. Questions and retrieved "
        "passages are sent to Gemini when you ask something. The index is kept in this "
        "session's memory and is cleared when the session is reset or expires. Do not use "
        "documents you are not authorized to process. A hosted app's configured Gemini "
        "key and quota are shared by its visitors."
    )

if selected_documents:
    with st.expander(f"Review selected documents · {len(selected_documents)}", expanded=True):
        prepared_pages = {}
        for source_name, content, content_id in selected_documents:
            prepared_pages[content_id] = _show_document_preview(
                source_name,
                content,
                content_id,
                use_ocr=(
                    use_ocr
                    and (
                        Path(source_name).suffix.lower() == ".pdf"
                        or Path(source_name).suffix.lower() in IMAGE_EXTENSIONS
                    )
                ),
                ocr_method=ocr_method,
            )

    # Keep the active index aligned with the documents the user selected.
    active_ids = {content_id for _name, _content, content_id in selected_documents}
    removed_ids = set(st.session_state["knowledge_sources"]) - active_ids
    if removed_ids:
        for content_id in removed_ids:
            st.session_state["knowledge_sources"].pop(content_id, None)
            st.session_state["indexed_extraction_modes"].pop(content_id, None)
            st.session_state["index_attempts"].pop(content_id, None)
            st.session_state["index_errors"].pop(content_id, None)
        st.session_state["chat_history"] = []

    if api_key:
        # Store only a one-way fingerprint to retry failed indexing when the key changes.
        key_fingerprint = _digest(api_key.encode("utf-8"))
        pending_documents = []
        pending_pages = {}
        pending_modes = {}
        for source_name, _content, content_id in selected_documents:
            suffix = Path(source_name).suffix.lower()
            requires_ocr = suffix == ".pdf" or suffix in IMAGE_EXTENSIONS
            extraction_mode = f"ocr:{ocr_method}" if use_ocr and requires_ocr else "text-extraction"
            previous_mode = st.session_state["indexed_extraction_modes"].get(content_id)
            if (
                content_id in st.session_state["knowledge_sources"]
                and previous_mode != extraction_mode
            ):
                st.session_state["knowledge_sources"].pop(content_id, None)
                st.session_state["indexed_extraction_modes"].pop(content_id, None)
                st.session_state["chat_history"] = []
            if content_id in st.session_state["knowledge_sources"]:
                continue
            if st.session_state["index_attempts"].get(content_id) == key_fingerprint:
                continue
            if suffix in IMAGE_EXTENSIONS and not use_ocr:
                continue
            if (
                use_ocr
                and requires_ocr
                and st.session_state["ocr_auto_modes"].get(content_id) != ocr_method
            ):
                continue
            pages = prepared_pages.get(content_id, [])
            if not pages or not any(page.get("text", "").strip() for page in pages):
                continue
            st.session_state["index_attempts"][content_id] = key_fingerprint
            st.session_state["index_errors"].pop(content_id, None)
            pending_documents.append((source_name, _content, content_id))
            pending_pages[content_id] = pages
            pending_modes[content_id] = extraction_mode

        new_index_count = 0
        if pending_documents:
            try:
                with st.spinner(
                    f"Indexing {len(pending_documents)} document(s) and preparing searchable passages..."
                ):
                    new_index_count, errors = _build_knowledge_base(
                        pending_documents,
                        pending_pages,
                        CHUNK_SIZE,
                        CHUNK_OVERLAP,
                        api_key,
                    )
                for _name, _content, content_id in pending_documents:
                    if content_id in st.session_state["knowledge_sources"]:
                        st.session_state["indexed_extraction_modes"][content_id] = pending_modes[
                            content_id
                        ]
                for content_id in pending_pages:
                    st.session_state["index_errors"].pop(content_id, None)
                for error in errors:
                    matching_id = next(
                        (
                            content_id
                            for name, _content, content_id in pending_documents
                            if error.startswith(f"{name}:")
                        ),
                        None,
                    )
                    if matching_id:
                        st.session_state["index_errors"][matching_id] = error
            except Exception as error:
                message = str(error)
                if "API_KEY_INVALID" in message or "API key not valid" in message:
                    message = (
                        "Google rejected this API key. Replace it in `.streamlit/secrets.toml` "
                        "with a valid Gemini API key, then restart the app."
                    )
                for _name, _content, content_id in pending_documents:
                    st.session_state["index_errors"][content_id] = message

        for content_id, message in st.session_state["index_errors"].items():
            if content_id in active_ids:
                st.error(f"Could not index a selected document: {message}")
        if new_index_count:
            st.rerun()
    elif selected_documents:
        st.info("Add your Gemini API key to index these documents.")

st.divider()
st.subheader("Ask your documents")
if not st.session_state["knowledge_sources"]:
    st.info("Upload a document to index it automatically and start asking questions.")

_render_history()

question = st.chat_input(
    "Ask a question about the indexed documents...",
    disabled=not bool(st.session_state["knowledge_sources"] and api_key),
)
if question:
    st.session_state["chat_history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.status("Working on your question…", expanded=True) as progress_status:
            result = _answer_question(
                question,
                MINIMUM_SCORE,
                TOP_K,
                api_key,
                progress=lambda message: progress_status.update(label=message),
            )
            progress_status.update(
                label="Answer ready. Checking citations…",
                state="complete",
                expanded=False,
            )
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
                "evidence": result["evidence"],
                "status": result["status"],
            }
        )
    except Exception as error:
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": f"I couldn't complete that request. Check the API key and connection, then try again. Details: {error}",
                "citations": [],
                "evidence": [],
                "status": "error",
            }
        )
    st.rerun()
