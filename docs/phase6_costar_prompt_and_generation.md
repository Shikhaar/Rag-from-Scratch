# Phase 6: CO-STAR Prompting Framework & Citation Grounding

Phase 6 addresses the generation boundary: synthesizing retrieved context into grounded, verifiable answers with strict anti-hallucination guardrails using the **CO-STAR** prompting methodology.

---

## 1. The CO-STAR Framework Breakdown

| Element | Description | Implementation in `COSTARPromptBuilder` |
|---|---|---|
| **C - Context** | Domain setting and background data | Verified excerpts extracted directly from ingested documents, labeled with index numbers, filenames, page numbers, and chunk IDs. |
| **O - Objective** | Concrete task assignment | Answer the user's inquiry strictly using the provided context, avoiding speculation. |
| **S - Style** | Formatting & structural conventions | Clear, analytical, structured technical prose using Markdown headers, lists, and bold terms. |
| **T - Tone** | Emotional timbre and certainty | Objective, precise, neutral, evidence-grounded, and authoritative. |
| **A - Audience** | Target persona level | Technical engineers and software practitioners who require factual accuracy. |
| **R - Response Format** | Explicit syntax and citation rules | Compulsory inline bracketed citations `[Source X]`, mandatory lack-of-knowledge disclosure if information is absent. |

---

## 2. Dynamic Citation Formatting

The `COSTARPromptBuilder` assembles retrieved search results into indexed source blocks:

```text
--- [Source 1: cloud_platform_policy.txt, Page N/A, Chunk 0] ---
Enterprise customers are eligible for a full refund within a 30-day trial period from the initial invoice date.

--- [Source 2: cloud_platform_policy.txt, Page N/A, Chunk 1] ---
To request a refund or credit, the organization's billing administrator must submit a ticket tagged 'Billing Dispute'.
```

### Citation Linking & Validation
- Every claim in the LLM answer is tagged with `[Source X]`.
- The `QueryResult` returns a structured citation catalog mapping each `[Source X]` directly back to the original document file path, page number, similarity score, and text preview.

---

## 3. Anti-Hallucination Guardrail

When an inquiry asks about facts outside the retrieved knowledge base, the system instructions enforce explicit disclosure:
> *"Based on the provided documents, I do not have enough information to answer this question."*

This prevents plausible-sounding fabrications and guarantees verifiable grounding.
