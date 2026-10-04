"""ColBERT late-interaction reranking.

ColBERT scores a query-document pair with token-level vectors instead of a
single document vector. RAGX keeps the default bi-encoder search path for
candidate recall, then applies this module to the fused candidate pool when
`RAG_RERANKER_BACKEND=colbert`.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag.interfaces import Document, ScoredDocument
from rag.retrieval.lexical import _tokenize


TokenVectors = list[list[float]]


class ColBERTEncoder(Protocol):
    """Token-level encoder used by the ColBERT reranker."""

    def encode_queries(self, queries: Sequence[str]) -> list[TokenVectors]:
        """Encode queries into token vectors.

        Example Input:
            encoder.encode_queries(["报销材料"])

        Example Output:
            [[[0.1, 0.2], [0.3, 0.4]]]
        """
        ...

    def encode_documents(self, documents: Sequence[Document]) -> list[TokenVectors]:
        """Encode documents into token vectors.

        Example Input:
            encoder.encode_documents([Document("a", "报销材料", {})])

        Example Output:
            [[[0.1, 0.2], [0.3, 0.4]]]
        """
        ...


@dataclass(frozen=True)
class ColBERTRerankConfig:
    """ColBERT reranking configuration."""

    model_name: str = ""
    query_max_tokens: int = 32
    document_max_tokens: int = 180
    batch_size: int = 8
    interaction_weight: float = 0.95
    retrieval_score_weight: float = 0.05
    hashing_dim: int = 64


class HashingColBERTEncoder:
    """Dependency-free ColBERT-shaped encoder for tests and offline validation."""

    def __init__(
        self,
        *,
        query_max_tokens: int = 32,
        document_max_tokens: int = 180,
        dimension: int = 64,
    ) -> None:
        """Initialize the deterministic local encoder.

        Example Input:
            HashingColBERTEncoder(query_max_tokens=8, document_max_tokens=32)

        Example Output:
            An encoder exposing ColBERT token-level vectors.
        """
        self._query_max_tokens = max(1, int(query_max_tokens))
        self._document_max_tokens = max(1, int(document_max_tokens))
        self._dimension = max(2, int(dimension))

    def encode_queries(self, queries: Sequence[str]) -> list[TokenVectors]:
        """Encode query strings.

        Example Input:
            encoder.encode_queries(["RAGX 检索"])

        Example Output:
            A list with one token-vector matrix per query.
        """
        return [
            self._encode_text(query, max_tokens=self._query_max_tokens)
            for query in queries
        ]

    def encode_documents(self, documents: Sequence[Document]) -> list[TokenVectors]:
        """Encode document content plus retrieval metadata.

        Example Input:
            encoder.encode_documents([Document("a", "content", {"source": "a.md"})])

        Example Output:
            A list with one token-vector matrix per document.
        """
        return [
            self._encode_text(
                _document_text(document),
                max_tokens=self._document_max_tokens,
            )
            for document in documents
        ]

    def _encode_text(self, text: str, *, max_tokens: int) -> TokenVectors:
        tokens = _unique(_tokenize(text))[:max_tokens]
        return [_token_vector(token, self._dimension) for token in tokens]


class TransformersColBERTEncoder:  # pragma: no cover - optional heavy dependency.
    """HuggingFace-backed ColBERT-style encoder.

    The implementation loads an AutoModel and uses normalized last hidden states
    as token vectors. It is intentionally optional so local tests do not need to
    download large models.
    """

    def __init__(
        self,
        model_name: str,
        *,
        query_max_tokens: int = 32,
        document_max_tokens: int = 180,
        batch_size: int = 8,
        device: str | None = None,
    ) -> None:
        """Load tokenizer and model.

        Example Input:
            TransformersColBERTEncoder("colbert-ir/colbertv2.0")

        Example Output:
            A token-level encoder backed by torch and transformers.
        """
        if not model_name.strip():
            raise ValueError("ColBERT model_name 不能为空")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ColBERT 模型加载需要可选依赖："
                "python3 -m pip install torch transformers"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._query_max_tokens = max(1, int(query_max_tokens))
        self._document_max_tokens = max(1, int(document_max_tokens))
        self._batch_size = max(1, int(batch_size))
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def encode_queries(self, queries: Sequence[str]) -> list[TokenVectors]:
        """Encode query strings.

        Example Input:
            encoder.encode_queries(["报销材料"])

        Example Output:
            A list with one token-vector matrix per query.
        """
        return self._encode_texts(queries, max_tokens=self._query_max_tokens)

    def encode_documents(self, documents: Sequence[Document]) -> list[TokenVectors]:
        """Encode documents.

        Example Input:
            encoder.encode_documents([Document("a", "报销材料", {})])

        Example Output:
            A list with one token-vector matrix per document.
        """
        texts = [_document_text(document) for document in documents]
        return self._encode_texts(texts, max_tokens=self._document_max_tokens)

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        max_tokens: int,
    ) -> list[TokenVectors]:
        results: list[TokenVectors] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start:start + self._batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
            }
            special_mask = encoded.pop("special_tokens_mask", None)
            with self._torch.no_grad():
                outputs = self._model(**encoded)
            hidden = self._torch.nn.functional.normalize(
                outputs.last_hidden_state,
                p=2,
                dim=-1,
            )
            valid_mask = encoded["attention_mask"].bool()
            if special_mask is not None:
                valid_mask = valid_mask & ~special_mask.bool()
            for token_embeddings, token_mask in zip(hidden, valid_mask):
                vectors = token_embeddings[token_mask].detach().cpu().tolist()
                results.append(vectors)
        return results


class ColBERTReranker:
    """Rerank fused candidates with ColBERT MaxSim interaction."""

    def __init__(
        self,
        encoder: ColBERTEncoder | None = None,
        config: ColBERTRerankConfig | None = None,
    ) -> None:
        """Initialize the ColBERT reranker.

        Example Input:
            ColBERTReranker()

        Example Output:
            A reranker exposing `rerank(query, contexts, top_k)`.
        """
        self._config = config or ColBERTRerankConfig()
        self._encoder = encoder or build_colbert_encoder(self._config)

    def rerank(
        self,
        query: str,
        contexts: Sequence[ScoredDocument],
        *,
        top_k: int,
    ) -> list[ScoredDocument]:
        """Run query-document token interaction and return sorted contexts.

        Example Input:
            reranker.rerank("报销材料", contexts, top_k=3)

        Example Output:
            A score-sorted list of ScoredDocument objects.
        """
        if top_k <= 0 or not contexts:
            return []

        documents = [context.document for context in contexts]
        query_vectors = self._encoder.encode_queries([query])[0]
        document_vectors = self._encoder.encode_documents(documents)
        raw_scores = _normalize_raw_scores([context.score for context in contexts])

        ranked: list[tuple[int, float, ScoredDocument]] = []
        for index, context in enumerate(contexts):
            interaction_score = _maxsim_score(
                query_vectors,
                document_vectors[index] if index < len(document_vectors) else [],
            )
            score = _combined_score(
                interaction_score,
                raw_scores[index],
                self._config,
            )
            ranked.append(
                (
                    index,
                    score,
                    ScoredDocument(document=context.document, score=score),
                )
            )

        ranked.sort(key=lambda item: (item[1], raw_scores[item[0]], -item[0]), reverse=True)
        return [item[2] for item in ranked[:top_k]]


def build_colbert_encoder(config: ColBERTRerankConfig) -> ColBERTEncoder:
    """Build a ColBERT encoder from configuration.

    Example Input:
        build_colbert_encoder(ColBERTRerankConfig(model_name=""))

    Example Output:
        HashingColBERTEncoder() when no external model is configured.
    """
    if config.model_name.strip():
        return TransformersColBERTEncoder(
            config.model_name,
            query_max_tokens=config.query_max_tokens,
            document_max_tokens=config.document_max_tokens,
            batch_size=config.batch_size,
        )
    return HashingColBERTEncoder(
        query_max_tokens=config.query_max_tokens,
        document_max_tokens=config.document_max_tokens,
        dimension=config.hashing_dim,
    )


def rank_by_colbert(
    query: str,
    contexts: Sequence[ScoredDocument],
    *,
    top_k: int,
    reranker: ColBERTReranker | None = None,
) -> list[ScoredDocument]:
    """Compatibility entrypoint for ColBERT reranking.

    Example Input:
        rank_by_colbert("报销", contexts, top_k=3)

    Example Output:
        A ColBERT-reranked context list.
    """
    return (reranker or ColBERTReranker()).rerank(query, contexts, top_k=top_k)


def _maxsim_score(query_vectors: TokenVectors, document_vectors: TokenVectors) -> float:
    """Compute ColBERT MaxSim interaction score.

    Example Input:
        _maxsim_score([[1.0, 0.0]], [[1.0, 0.0]])

    Example Output:
        1.0
    """
    if not query_vectors or not document_vectors:
        return 0.0
    scores = [
        max(_cosine(query_vector, doc_vector) for doc_vector in document_vectors)
        for query_vector in query_vectors
    ]
    normalized = (sum(scores) / len(scores) + 1.0) / 2.0
    return round(_clamp(normalized), 6)


def _combined_score(
    interaction_score: float,
    raw_score: float,
    config: ColBERTRerankConfig,
) -> float:
    score = (
        _clamp(interaction_score) * config.interaction_weight
        + _clamp(raw_score) * config.retrieval_score_weight
    )
    return round(_clamp(score), 6)


def _normalize_raw_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [1.0 if score > 0 else 0.0 for score in scores]
    span = maximum - minimum
    return [_clamp((score - minimum) / span) for score in scores]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


def _token_vector(token: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        payload = f"{token}:{counter}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=32).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    return _l2_normalize(values[:dimension])


def _l2_normalize(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _document_text(document: Document) -> str:
    metadata = document.metadata
    parts = [
        str(document.content or ""),
        _metadata_text(metadata.get("heading_path")),
        _metadata_text(metadata.get("source")),
        _metadata_text(metadata.get("doc_id")),
    ]
    return " ".join(part for part in parts if part).strip()


def _metadata_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item is not None)
    if value is None:
        return ""
    return str(value)


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
