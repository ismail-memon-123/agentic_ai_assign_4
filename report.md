# RAG System Evaluation Report

## Overview

This report summarizes the evaluation of the Gemini Retrieval-Augmented Generation (RAG) system. The evaluation measured retrieval accuracy, response quality, JSON formatting reliability, and search performance.

---

# Test Configuration

| Metric | Value |
|---------|------:|
| LLM | Google Gemini |
| Embedding Model | Gemini Embeddings |
| Vector Database | FAISS |
| Retrieval Method | IVF-PQ |
| Total Queries | 17 |
| Response Format | Structured JSON (Pydantic Validated) |

---

# Evaluation Results

## Negative Question Evaluation

Negative questions were intentionally designed so that the answer did **not** exist within the indexed corpus. The expected behavior was for the system to respond that there was insufficient information.

| Metric | Result |
|---------|-------:|
| Negative Questions | 5 |
| Correct "Not Enough Information" Responses | 5 |
| Incorrect Responses | 0 |
| Accuracy | **100%** |

The system correctly refused to fabricate information for every negative query, demonstrating successful grounding in the retrieved context.

---

## Positive Question Evaluation

Positive questions contained information present within the indexed documents.

| Metric | Result |
|---------|-------:|
| Positive Questions | 12 |
| Correct Answers | 8 |
| Incorrect Responses | 4 |
| Accuracy | **66.7%** |

Breakdown of correct responses:

| Retrieval Type | Count |
|----------------|------:|
| Single Source Citation | 3 |
| Multiple Source Citations | 5 |

The incorrect responses consisted primarily of the model returning **"I don't have enough information"** despite the required information existing within the indexed documents. This behavior is conservative and is likely attributable to retrieval limitations or prompt grounding rather than hallucinated content.

---

# Overall Accuracy

| Metric | Value |
|---------|------:|
| Total Queries | 17 |
| Correct Responses | 13 |
| Incorrect Responses | 4 |
| Overall Accuracy | **76.5%** |

---

# JSON Output Validation

All Gemini responses were required to conform to the predefined Pydantic schema.

| Metric | Result |
|---------|-------:|
| Total Responses | 17 |
| Valid JSON on First Attempt | 17 |
| Validation Success Rate | **100%** |
| Retry Required | 0 |

Every generated response satisfied the required JSON schema on the initial generation attempt.

---

# Search Performance

Semantic retrieval performance was measured using the elapsed FAISS search time for each query.

| Metric | Value |
|---------|------:|
| Average Search Time | **290 µs (0.29 ms)** |

The FAISS index provided low-latency retrieval suitable for interactive question answering.

---

# Summary

| Evaluation Metric | Result |
|-------------------|--------|
| Negative Question Accuracy | **100%** |
| Positive Question Accuracy | **66.7%** |
| Overall Accuracy | **76.5%** |
| JSON Validation Success | **100%** |
| Average Search Time | **290 µs** |

---

# Discussion

The evaluation demonstrates that the RAG pipeline reliably generates grounded responses while avoiding hallucinations for questions whose answers are absent from the indexed corpus. The system correctly identified all negative test cases, indicating effective use of retrieval-based grounding.

For positive queries, approximately two-thirds of responses correctly identified the relevant information. The remaining incorrect responses were conservative failures ("not enough information") rather than fabricated answers, suggesting opportunities to improve retrieval performance through tuning parameters such as chunk size, overlap, Top-K retrieval, or FAISS IVF-PQ configuration.

Overall, the system achieved reliable structured output, low retrieval latency, and strong grounding behavior, satisfying the primary objectives of the production-oriented RAG assignment.
