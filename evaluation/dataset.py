"""
Evaluation dataset schema and persistence for RAG benchmarking.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class EvalSample:
    """A single evaluation query sample with ground truth labels."""
    query: str
    relevant_doc_names: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevant_keywords: List[str] = field(default_factory=list)
    ground_truth_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalDataset:
    """A collection of evaluation samples for benchmark testing."""
    name: str
    description: str = ""
    samples: List[EvalSample] = field(default_factory=list)

    def add_sample(self, sample: EvalSample) -> None:
        self.samples.append(sample)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "name": self.name,
            "description": self.description,
            "samples": [
                {
                    "query": s.query,
                    "relevant_doc_names": s.relevant_doc_names,
                    "relevant_chunk_ids": s.relevant_chunk_ids,
                    "relevant_keywords": s.relevant_keywords,
                    "ground_truth_answer": s.ground_truth_answer,
                    "metadata": s.metadata,
                }
                for s in self.samples
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EvalDataset":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(name=data.get("name", "dataset"), description=data.get("description", ""))
        for s in data.get("samples", []):
            dataset.samples.append(
                EvalSample(
                    query=s["query"],
                    relevant_doc_names=s.get("relevant_doc_names", []),
                    relevant_chunk_ids=s.get("relevant_chunk_ids", []),
                    relevant_keywords=s.get("relevant_keywords", []),
                    ground_truth_answer=s.get("ground_truth_answer"),
                    metadata=s.get("metadata", {}),
                )
            )
        return dataset
