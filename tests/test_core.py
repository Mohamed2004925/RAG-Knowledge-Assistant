"""Fast, offline tests for the RAG project's core contracts."""

import tempfile
import unittest
from pathlib import Path

from chunking import chunk_pages
from evaluation import score_evaluation_cases
from generation import validate_citations
from retrieval import cosine_similarity, rank_chunks
from vector_store import delete_source, load_chunks, search_index, upsert_document


class ChunkingTests(unittest.TestCase):
    def test_chunks_keep_page_and_source_metadata(self):
        pages = [{"label": "Page 1", "text": "A useful policy sentence. " * 12}]

        chunks = chunk_pages(pages, "policy.pdf", chunk_size=80, overlap=15)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["source"] == "policy.pdf" for chunk in chunks))
        self.assertTrue(all(chunk["page"] == "Page 1" for chunk in chunks))
        self.assertTrue(all(chunk["start_char"] < chunk["end_char"] for chunk in chunks))

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_pages([{"label": "Page 1", "text": "text"}], "policy.pdf", 20, 20)


class RetrievalTests(unittest.TestCase):
    def test_cosine_similarity_and_ranking(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.5, 0.0]), 1.0)
        chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]

        ranked = rank_chunks([1.0, 0.0], chunks, [[0.0, 1.0], [1.0, 0.0]], top_k=1)

        self.assertEqual(ranked[0]["chunk"]["chunk_id"], "b")
        self.assertAlmostEqual(ranked[0]["score"], 1.0)

    def test_rejects_vectors_with_different_dimensions(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])


class CitationTests(unittest.TestCase):
    def test_accepts_only_request_local_citation_ids(self):
        valid = validate_citations("The policy says so [S1].", {"S1"})
        invented = validate_citations("The policy says so [S2].", {"S1"})
        missing = validate_citations("The policy says so.", {"S1"})

        self.assertEqual(valid["status"], "answered")
        self.assertEqual(valid["citations"], ["S1"])
        self.assertEqual(invented["status"], "citation_error")
        self.assertEqual(missing["status"], "citation_error")

    def test_maps_abstention_marker_to_fallback(self):
        result = validate_citations("INSUFFICIENT_EVIDENCE", {"S1"})

        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["citations"], [])


class EvaluationTests(unittest.TestCase):
    def test_scores_answerability_retrieval_and_citation_pages(self):
        cases = [
            {
                "question": "Where is the rule?",
                "expected_answerable": True,
                "expected_source": "policy.pdf",
                "expected_page": "Page 2",
            }
        ]
        chunk = {"source": "policy.pdf", "page": "Page 2"}
        runs = [
            {
                "answer_status": "answered",
                "answer": "It is on page 2 [S1].",
                "retrieved": [{"chunk": chunk}],
                "cited_chunks": [chunk],
            }
        ]

        summary = score_evaluation_cases(cases, runs)

        self.assertEqual(summary["answerability_accuracy"], 1.0)
        self.assertEqual(summary["retrieval_hit_rate"], 1.0)
        self.assertEqual(summary["citation_hit_rate"], 1.0)


class VectorStoreTests(unittest.TestCase):
    def test_index_search_and_delete_are_persistent(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        db_path = Path(temporary_directory.name) / "test_index.sqlite3"
        chunks = [
            {
                "chunk_id": "one",
                "page": "Page 1",
                "chunk_number": 1,
                "start_char": 0,
                "end_char": 12,
                "text": "vacation rule",
            }
        ]

        upsert_document(
            "source-a",
            "policy.pdf",
            "content-hash",
            chunks,
            [[1.0, 0.0]],
            "test-embedding-model",
            db_path,
        )
        loaded = load_chunks(db_path=db_path)
        results = search_index([1.0, 0.0], top_k=1, db_path=db_path)

        self.assertEqual(loaded[0]["page"], "Page 1")
        self.assertEqual(results[0]["chunk"]["source"], "policy.pdf")
        self.assertAlmostEqual(results[0]["score"], 1.0)

        delete_source("source-a", db_path=db_path)
        self.assertEqual(load_chunks(db_path=db_path), [])


if __name__ == "__main__":
    unittest.main()
