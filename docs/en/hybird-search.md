# RAGX Hybrid Search Troubleshooting Guide

Retrieval-Augmented Generation (RAG) hybrid retrieval usually means dense vector retrieval + sparse/keyword retrieval, followed by fusion and reranking. In practice, problems rarely come from "the model not being strong enough" but from engineering details and data handling. The sections below go through common pitfalls at each stage of the retrieval pipeline.

## Abbreviations

| Abbreviation | Full Name | Description |
| --- | --- | --- |
| BM25 | Best Matching 25 | Lexical relevance ranking function, used for keyword recall. |
| HyDE | Hypothetical Document Embeddings | Improves vector representation of short queries by generating hypothetical documents. |
| LLM | Large Language Model | Large language model, responsible for final answer generation. |
| RAG | Retrieval-Augmented Generation | Retrieval-augmented generation. |
| RRF | Reciprocal Rank Fusion | Rank-based fusion method for multi-channel recall. |

For the full abbreviation table, see [RAGX Abbreviations](abbreviations.md).

## 1. Chunking Stage

- **Chunk granularity mismatched to query granularity**: Chunks that are too large recall more noise, dilute relevant signals, and overflow the reranking context; chunks that are too small are semantically incomplete, and answers get cut across two chunks. Prefer splitting by semantics/structure (heading hierarchy, paragraphs) rather than hard-cutting at a fixed character count.
- **Lost context anchors**: Stripping tables, lists, or code from their headings and preceding context leaves the chunk without a referent (e.g. "this metric" or "the table above" becomes uninterpretable). Common fixes are attaching metadata such as parent headings and document titles to the chunk, or using small-to-big / parent-document retrieval.
- **Improper overlap settings**: No overlap loses boundary information; too much overlap inflates the index and causes duplicate recalls.

## 2. Sparse Retrieval Stage (Best Matching 25, BM25/Keywords)

- **Missing tokenization and language processing**: Chinese not properly tokenized, stopwords and casing not handled, rendering BM25 effectively useless. Mixed Chinese-English text, proper nouns, and code identifiers are especially prone to pitfalls here.
- **Synonym/terminology gap**: Users write colloquial terms while documents use technical terms, so pure lexical matching recalls nothing — this is exactly the gap dense retrieval is supposed to fill, but setting the sparse-side weight too high re-amplifies the problem.

## 3. Dense Retrieval Stage (Vectors)

- **Embedding model mismatched to the domain**: General-purpose embeddings discriminate poorly in vertical domains (legal, medical, internal jargon). Always evaluate with business data and fine-tune when necessary.
- **Query-document asymmetry**: Short questions and long document chunks differ greatly in vector space distribution. Use a model supporting asymmetric retrieval, or narrow the distribution gap with query rewriting/Hypothetical Document Embeddings (HyDE).
- **Inconsistent normalization and distance metric**: Indexing with inner product while querying with cosine, or skipping normalization, scrambles scoring.
- **Multilingual/cross-lingual scenarios**: Using a monolingual model while needing cross-lingual retrieval.

## 4. Fusion and Rerank Stage — the Most Error-Prone Stage

- **Adding scores directly**: Best Matching 25 (BM25) scores and vector similarity have completely different scales; directly weighted fusion lets one channel dominate the results. **Prefer Reciprocal Rank Fusion (RRF), which fuses based on ranks rather than raw scores**; it is insensitive to scale and more robust.
- **Fusion weights set by gut feeling**: Dense/sparse weights (alpha) never tuned on a real query set, arbitrarily set to 0.5. Optimal weights differ across query types (factual vs. semantic).
- **Skipping two-stage reranking**: After hybrid recall, truncating Top-K by fusion score alone leaves fine ranking missing, and truly relevant documents never surface. RAGX can switch between Cross-Encoder, ColBERT, and no reranking via `RAG_RERANKER_BACKEND=cross_encoder|colbert|none`.
- **Reranking candidate set too small**: Recall K before reranking set too small (e.g. only 5 per channel) means good documents are already missed at the recall stage, and no reranker can recover.

## 5. Retrieval-to-Generation Handoff Stage

- **Top-K imbalanced against the context window**: Stuffing too many chunks into the prompt makes the Large Language Model (LLM) "lost in the middle", ignoring key information in the middle of the long context.
- **Missing relevance threshold/fallback**: Feeding everything to the Large Language Model (LLM) regardless of retrieval quality still generates answers when the knowledge base has none, causing hallucinations. Set a score threshold and answer "not found" below it.
- **No citation traceability**: Without knowing which chunk an answer came from, verification and debugging become hard.

## 6. System and Evaluation Level

- **No per-stage evaluation**: Looking only at end-to-end accuracy makes it impossible to tell whether recall missed, reranking misordered, or generation hallucinated. Monitor **retrieval recall, reranking hit rate, and generation faithfulness** separately.
- **Lacking a real evaluation set**: Tuning on synthetic questions does not match the distribution of real online queries.
- **Metadata filtering missing or too strict**: No permission/time/category filtering recalls unauthorized or stale content; overly strict filters cut the candidate set to nothing.
- **Index updates and consistency**: After document updates, the vector store and sparse index are not rebuilt in sync, so the two retrieval channels see different versions of the data.
- **Cost and latency**: Dual-channel retrieval + Cross-Encoder reranking significantly increases latency and compute cost; under high concurrency, use caching, asynchronization, or reranker distillation.
