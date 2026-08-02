"""RAG: Chroma vector store + cross-encoder reranker + citations.

The pipeline:
  1. build_index(): chunk each KB doc, embed with a MiniLM bi-encoder, and store in
     Chroma. Idempotent — the collection is dropped and rebuilt on each call.
  2. retrieve(query): fetch `fetch_k` candidates from Chroma, then rerank them with a
     cross-encoder and keep the top `k`. The cross-encoder scores each (query,
     passage) pair together, which is more accurate than cosine similarity alone.
  3. Each result keeps its doc_id, which becomes the [citation] shown to the user.

Models are loaded lazily and cached at module level, so the reranker is only built
the first time it's used.
"""
from __future__ import annotations

import re

import chromadb
from chromadb.utils import embedding_functions

from . import config
from .kb_docs import KB_DOCS

_client = None
_collection = None
_reranker = None
_embed_fn = None


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
    return _embed_fn


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=_get_embed_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


def _chunk(text: str, max_chars: int = 350) -> list[str]:
    """Group sentences into chunks of at most ~max_chars.

    The docs are short, so a sentence-aware greedy grouping gives 1-3 focused
    chunks per doc — enough granularity for retrieval without shredding context.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > max_chars:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf.strip())
    return chunks


def build_index() -> int:
    """Drop and rebuild the Chroma collection from KB_DOCS. Returns chunk count."""
    client = _get_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    # Force the module-level handle to re-create the (now empty) collection.
    global _collection
    _collection = None
    col = _get_collection()

    ids, docs, metas = [], [], []
    for doc in KB_DOCS:
        for i, chunk in enumerate(_chunk(doc["text"])):
            ids.append(f'{doc["id"]}::{i}')
            docs.append(chunk)
            metas.append({"doc_id": doc["id"], "title": doc["title"], "chunk": i})
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def retrieve(query: str, k: int = 3, fetch_k: int = 8) -> list[dict]:
    """Return top-k reranked passages for `query`, each with its citation handle."""
    col = _get_collection()
    res = col.query(query_texts=[query], n_results=fetch_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    if not docs:
        return []

    scores = _get_reranker().predict([(query, d) for d in docs])
    ranked = sorted(
        zip(scores, docs, metas), key=lambda x: float(x[0]), reverse=True
    )[:k]
    return [
        {
            "doc_id": m["doc_id"],
            "title": m["title"],
            "text": d,
            "score": round(float(s), 4),
        }
        for s, d, m in ranked
    ]


if __name__ == "__main__":
    # Quick check: `python -m helppilot.rag "my package never arrived"`
    import sys

    q = " ".join(sys.argv[1:]) or "my package never arrived, can I get a refund?"
    print(f"Query: {q}\n")
    for r in retrieve(q):
        print(f"[{r['doc_id']}] ({r['score']})  {r['title']}")
        print(f"    {r['text'][:160]}...\n")
