"""
Prompt engineering engine using the CO-STAR (Context, Objective, Style, Tone, Audience, Response) framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from core.schema import SearchResult


@dataclass
class COSTARConfig:
    """
    Configuration dataclass for the CO-STAR Prompting Framework.
    
    C - Context: Background knowledge and domain setting.
    O - Objective: The precise task and answering goals.
    S - Style: Writing structure, formatting, and analytical depth.
    T - Tone: Emotional timbre, objectivity, and certainty level.
    A - Audience: Target reader expertise level.
    R - Response: Specific formatting rules, citation syntax, and schema constraints.
    """
    context_preamble: str = (
        "You are an expert, highly reliable retrieval-augmented assistant. "
        "You have access to verified reference excerpts extracted directly from authoritative documentation."
    )
    objective: str = (
        "Answer the user's inquiry with 100% factual fidelity based exclusively on the provided Context Sources. "
        "Do not extrapolate, assume, or hallucinate beyond what is explicitly stated in the reference text."
    )
    style: str = "Clear, analytical, structured technical prose using Markdown headers, lists, and bold key terms where appropriate."
    tone: str = "Objective, precise, neutral, and authoritative."
    audience: str = "Technical engineers, software architects, and domain professionals who value verifiable factual accuracy."
    response_format: str = (
        "1. Every factual statement or claim MUST include an inline citation tag: [Source X] (e.g. 'The refund window is 30 days [Source 1].').\n"
        "2. If multiple sources corroborate a claim, combine citations: [Source 1, Source 2].\n"
        "3. If the provided Context Sources do NOT contain enough information to answer the question, state: "
        "'Based on the provided documents, I do not have enough information to answer this question.'\n"
        "4. Never mention source documents that were not provided in the Context Sources."
    )


class BasePromptBuilder(ABC):
    """Abstract Base Class for prompt formatting engines."""

    @abstractmethod
    def format_context(self, results: List[SearchResult]) -> Tuple[str, List[Dict[str, Any]]]:
        """Format retrieved search results into a clean context block with citation metadata."""
        pass

    @abstractmethod
    def build_prompt(
        self, query: str, results: List[SearchResult]
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Build (system_prompt, user_prompt, citations) tuple."""
        pass


class COSTARPromptBuilder(BasePromptBuilder):
    """
    Implementation of the CO-STAR Prompt Engineering Framework.
    Constructs highly grounded, citation-aware prompts for Large Language Models.
    """

    def __init__(self, config: Optional[COSTARConfig] = None):
        self.config = config or COSTARConfig()

    def _build_system_prompt(self) -> str:
        """Synthesize the system instruction using the CO-STAR structure."""
        return f"""[CO-STAR INSTRUCTION FRAMEWORK]

# CONTEXT
{self.config.context_preamble}

# OBJECTIVE
{self.config.objective}

# STYLE
{self.config.style}

# TONE
{self.config.tone}

# AUDIENCE
{self.config.audience}

# RESPONSE FORMAT & CITATION RULES
{self.config.response_format}"""

    def format_context(self, results: List[SearchResult]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Format retrieved search results into structured context blocks with citation mappings.
        """
        context_parts = []
        citations = []

        for idx, res in enumerate(results, 1):
            source = res.metadata.get("filename", res.metadata.get("source", "Unknown"))
            page = res.metadata.get("page")
            chunk_idx = res.chunk.chunk_index

            header_info = f"Source {idx}: {source}"
            if page is not None:
                header_info += f", Page {page}"
            header_info += f", Chunk {chunk_idx}"

            citation_entry = {
                "source_id": idx,
                "label": f"[Source {idx}]",
                "source": source,
                "full_path": res.metadata.get("source", ""),
                "page": page,
                "chunk_index": chunk_idx,
                "chunk_id": res.chunk.chunk_id,
                "score": res.score,
                "rerank_score": res.rerank_score,
                "dense_score": res.dense_score,
                "bm25_score": res.bm25_score,
                "preview": res.chunk.content[:150] + ("..." if len(res.chunk.content) > 150 else ""),
            }
            citations.append(citation_entry)

            block = f"--- [{header_info}] ---\n{res.chunk.content.strip()}"
            context_parts.append(block)

        formatted_context = "\n\n".join(context_parts) if context_parts else "No relevant context sources found."
        return formatted_context, citations

    def build_prompt(
        self, query: str, results: List[SearchResult]
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """
        Assemble system instruction and user prompt according to CO-STAR specifications.
        """
        system_prompt = self._build_system_prompt()
        context_str, citations = self.format_context(results)

        user_prompt = f"""# CONTEXT SOURCES:
======================================================================
{context_str}
======================================================================

# USER QUESTION:
{query}

Please formulate your grounded answer adhering strictly to the CO-STAR response and citation specifications above."""

        return system_prompt, user_prompt, citations


# Backward-compatible alias for existing code
PromptBuilder = COSTARPromptBuilder
