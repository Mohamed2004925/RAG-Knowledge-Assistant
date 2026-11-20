"""Small retrieval and abstention metrics for Chapter 7."""

import hashlib
import json


def evaluation_signature(
    cases: list[dict],
    source_ids: list[str],
    top_k: int,
    minimum_score: float,
) -> str:
    """Identify the exact evaluation inputs so stale results are not reused."""
    normalized_cases = [
        {
            "question": str(case.get("question", "")).strip(),
            "expected_answerable": bool(case.get("expected_answerable", False)),
            "expected_source": str(case.get("expected_source", "")).strip(),
            "expected_page": str(case.get("expected_page", "")).strip(),
        }
        for case in cases
    ]
    payload = {
        "cases": normalized_cases,
        "source_ids": sorted(source_ids),
        "top_k": top_k,
        "minimum_score": minimum_score,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def score_evaluation_cases(cases: list[dict], runs: list[dict]) -> dict:
    """Calculate retrieval page hit, answerability accuracy, and citation page hit."""
    if len(cases) != len(runs):
        raise ValueError("Every evaluation case must have exactly one run result.")
    scored = []
    for case, run in zip(cases, runs):
        expected_answerable = bool(case["expected_answerable"])
        answered = run["answer_status"] == "answered"
        answerability_correct = answered == expected_answerable

        expected_page = str(case.get("expected_page", "")).strip()
        expected_source = str(case.get("expected_source", "")).strip()
        retrieved = run["retrieved"]
        if expected_page:
            retrieval_hit = any(
                match["chunk"]["page"] == expected_page
                and (not expected_source or match["chunk"]["source"] == expected_source)
                for match in retrieved
            )
        else:
            retrieval_hit = None

        cited = run.get("cited_chunks", [])
        if expected_answerable and expected_page:
            citation_page_hit = any(
                chunk["page"] == expected_page
                and (not expected_source or chunk["source"] == expected_source)
                for chunk in cited
            )
        else:
            citation_page_hit = None

        scored.append(
            {
                "question": case["question"],
                "expected": "answer" if expected_answerable else "abstain",
                "actual": "answer" if answered else "abstain",
                "answerability_correct": answerability_correct,
                "expected_page": expected_page,
                "retrieved_pages": ", ".join(
                    dict.fromkeys(match["chunk"]["page"] for match in retrieved)
                ),
                "retrieval_hit": retrieval_hit,
                "citation_page_hit": citation_page_hit,
                "status": run["answer_status"],
                "answer": run["answer"],
            }
        )

    answerability_accuracy = (
        sum(row["answerability_correct"] for row in scored) / len(scored)
        if scored
        else None
    )
    retrieval_cases = [row for row in scored if row["retrieval_hit"] is not None]
    retrieval_hit_rate = (
        sum(row["retrieval_hit"] for row in retrieval_cases) / len(retrieval_cases)
        if retrieval_cases
        else None
    )
    citation_cases = [row for row in scored if row["citation_page_hit"] is not None]
    citation_hit_rate = (
        sum(row["citation_page_hit"] for row in citation_cases) / len(citation_cases)
        if citation_cases
        else None
    )
    return {
        "rows": scored,
        "answerability_accuracy": answerability_accuracy,
        "retrieval_hit_rate": retrieval_hit_rate,
        "citation_hit_rate": citation_hit_rate,
        "case_count": len(scored),
    }
