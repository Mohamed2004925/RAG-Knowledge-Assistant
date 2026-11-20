"""A small character-based chunker for learning how retrieval units are formed."""

import re


def chunk_pages(
    pages: list[dict[str, str]],
    source_name: str,
    chunk_size: int = 300,
    overlap: int = 60,
) -> list[dict[str, str | int]]:
    """Split each page into word-boundary chunks and retain source metadata.

    Character offsets refer to normalized page text. Pages are chunked separately
    so a chunk never loses its page-level citation source.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks: list[dict[str, str | int]] = []
    for page in pages:
        paragraphs = [
            re.sub(r"\s+", " ", paragraph).strip()
            for paragraph in re.split(r"\n\s*\n", page.get("text", ""))
            if paragraph.strip()
        ]
        text = "\n\n".join(paragraphs)
        if not text:
            continue

        start = 0
        chunk_number = 1
        while start < len(text):
            proposed_end = min(start + chunk_size, len(text))
            end = proposed_end
            if proposed_end < len(text):
                paragraph_boundary = text.rfind("\n\n", start + 1, proposed_end)
                word_boundary = text.rfind(" ", start + 1, proposed_end)
                boundary = max(paragraph_boundary + 2, word_boundary)
                if boundary > start + chunk_size // 2:
                    end = boundary
                else:
                    # A nearby boundary would make a tiny chunk; extend to the next word.
                    following_word_boundary = text.find(" ", proposed_end)
                    end = following_word_boundary if following_word_boundary >= 0 else len(text)

            chunk_start = start
            while chunk_start < end and text[chunk_start].isspace():
                chunk_start += 1
            chunk_end = end
            while chunk_end > chunk_start and text[chunk_end - 1].isspace():
                chunk_end -= 1

            chunk_text = text[chunk_start:chunk_end]
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{source_name}::{page['label']}::chunk-{chunk_number}",
                        "source": source_name,
                        "page": page["label"],
                        "chunk_number": chunk_number,
                        "start_char": chunk_start,
                        "end_char": chunk_end,
                        "text": chunk_text,
                    }
                )
                chunk_number += 1

            if end >= len(text):
                break

            next_start = max(start + 1, end - overlap)
            while next_start < len(text) and not text[next_start].isspace():
                next_start += 1
            while next_start < len(text) and text[next_start].isspace():
                next_start += 1
            start = next_start

    return chunks
