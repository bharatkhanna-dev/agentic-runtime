from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ResearchCase(BaseModel):
    case_id: str
    question: str
    required_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    expected_citation_ids: list[str] = Field(default_factory=list)
    max_retrieved_documents: int = 4


class ResearchPrediction(BaseModel):
    answer: str
    retrieved_document_ids: list[str] = Field(default_factory=list)
    cited_document_ids: list[str] = Field(default_factory=list)


class ResearchScore(BaseModel):
    keyword_coverage: float
    forbidden_keyword_violations: int
    citation_recall: float
    retrieval_budget_respected: bool
    success: bool


def load_research_cases(file_path: str | Path) -> list[ResearchCase]:
    raw_cases = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return [ResearchCase.model_validate(item) for item in raw_cases]


def score_research_case(case: ResearchCase, prediction: ResearchPrediction) -> ResearchScore:
    normalized_answer = prediction.answer.lower()
    matched_keywords = sum(keyword.lower() in normalized_answer for keyword in case.required_keywords)
    keyword_coverage = matched_keywords / len(case.required_keywords) if case.required_keywords else 1.0

    forbidden_keyword_violations = sum(
        keyword.lower() in normalized_answer for keyword in case.forbidden_keywords
    )

    expected_citations = set(case.expected_citation_ids)
    actual_citations = set(prediction.cited_document_ids)
    citation_recall = (
        len(expected_citations & actual_citations) / len(expected_citations)
        if expected_citations
        else 1.0
    )

    retrieval_budget_respected = len(prediction.retrieved_document_ids) <= case.max_retrieved_documents
    success = (
        keyword_coverage == 1.0
        and forbidden_keyword_violations == 0
        and citation_recall == 1.0
        and retrieval_budget_respected
    )
    return ResearchScore(
        keyword_coverage=keyword_coverage,
        forbidden_keyword_violations=forbidden_keyword_violations,
        citation_recall=citation_recall,
        retrieval_budget_respected=retrieval_budget_respected,
        success=success,
    )


def summarize_research_scores(scores: list[ResearchScore]) -> dict[str, Any]:
    total = len(scores)
    if total == 0:
        return {
            "total_cases": 0,
            "task_success_rate": 0.0,
            "mean_keyword_coverage": 0.0,
            "citation_recall": 0.0,
            "retrieval_budget_respected_rate": 0.0,
        }

    return {
        "total_cases": total,
        "task_success_rate": sum(score.success for score in scores) / total,
        "mean_keyword_coverage": sum(score.keyword_coverage for score in scores) / total,
        "citation_recall": sum(score.citation_recall for score in scores) / total,
        "retrieval_budget_respected_rate": sum(score.retrieval_budget_respected for score in scores) / total,
    }
