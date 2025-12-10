"""Hybrid retrieval with dense + BM25 + RRF + BGE rerank."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path so that `utils.*` works when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import chromadb
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from tiktoken import get_encoding

from utils.prompt_config import (
    build_messages_answer,
    build_messages_judge,
)
from utils.llm_answer import generate_answer


def load_chunks(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def dense_search(
    collection, query_vec: List[float], k: int
) -> List[Tuple[str, float, Dict, str]]:
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc_id, meta, dist, doc in zip(
        res["ids"][0], res["metadatas"][0], res["distances"][0], res["documents"][0]
    ):
        # Convert distance to score; smaller dist -> higher score
        score = 1.0 / (1.0 + dist)
        hits.append((doc_id, score, meta, doc))
    return hits


def build_bm25(chunks: List[dict]) -> BM25Retriever:
    docs = [
        Document(page_content=d["text"], metadata=d)
        for d in chunks
        if d.get("text", "").strip()
    ]
    retriever = BM25Retriever.from_documents(docs)
    return retriever


def sparse_search(
    retriever: BM25Retriever, query: str, k: int
) -> List[Tuple[str, float, Dict, str]]:
    # 兼容当前版本：使用 runnable 接口 invoke 获取文档
    docs = retriever.invoke(query)
    hits = []
    for i, doc in enumerate(docs[:k]):
        meta = doc.metadata
        doc_id = f"bm25-{meta.get('source','')}-{meta.get('block_idx', i)}"
        hits.append((doc_id, float(k - i), meta, doc.page_content))
    return hits


def rrf_fuse(
    dense_hits: List[Tuple[str, float, Dict, str]],
    sparse_hits: List[Tuple[str, float, Dict, str]],
    k: int = 60,
    top_n: int = 50,
) -> List[Tuple[str, float, Dict, str]]:
    pool: Dict[str, Tuple[float, Dict, str]] = {}
    for hits in (dense_hits, sparse_hits):
        for rank, (doc_id, score, meta, doc) in enumerate(hits, start=1):
            contrib = 1.0 / (k + rank)
            if doc_id not in pool:
                pool[doc_id] = (contrib, meta, doc)
            else:
                pool[doc_id] = (pool[doc_id][0] + contrib, meta, doc)
    fused = sorted(pool.items(), key=lambda x: x[1][0], reverse=True)
    return [(doc_id, sc_meta_doc[0], sc_meta_doc[1], sc_meta_doc[2]) for doc_id, sc_meta_doc in fused[:top_n]]


def rerank_bge(
    reranker: FlagReranker,
    query: str,
    candidates: List[Tuple[str, float, Dict, str]],
    top_n: int = 8,
) -> List[Tuple[str, float, Dict, str]]:
    if not candidates:
        return []
    pairs = [[query, doc] for _, _, _, doc in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    reranked = []
    for (doc_id, _, meta, doc), sc in zip(candidates, scores):
        reranked.append((doc_id, float(sc), meta, doc))
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_n]


def dedup_results(
    items: List[Tuple[str, float, Dict, str]]
) -> List[Tuple[str, float, Dict, str]]:
    seen = set()
    uniq = []
    for doc_id, sc, meta, doc in items:
        key = (meta.get("source", ""), meta.get("block_idx", -1))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((doc_id, sc, meta, doc))
    return uniq


def load_chunk_map(chunks: List[dict]) -> Dict[Tuple[str, int], dict]:
    mp: Dict[Tuple[str, int], dict] = {}
    for c in chunks:
        key = (c["source"], int(c["block_idx"]))
        c["chunk_id"] = f"{c['source']}:{int(c['block_idx'])}"
        mp[key] = c
    return mp


def collect_with_neighbors(
    top_hits: List[Tuple[str, float, Dict, str]],
    chunk_map: Dict[Tuple[str, int], dict],
    radius: int = 1,
    max_tokens: int = 1500,
    encoder_name: str = "cl100k_base",
) -> List[Tuple[str, str]]:
    """Collect hit chunks with neighbor blocks in the same section, limited by tokens."""
    enc = get_encoding(encoder_name)
    seen = set()
    collected: List[Tuple[str, str]] = []
    total_tokens = 0
    for _, _, meta, _doc in top_hits:
        src = meta.get("source", "")
        bi = int(meta.get("block_idx", -1))
        sec = meta.get("section_path", "")
        for nb in range(bi - radius, bi + radius + 1):
            key = (src, nb)
            if key in seen:
                continue
            chunk = chunk_map.get(key)
            if not chunk:
                # Fallback: try string key if any
                chunk = chunk_map.get((src, int(nb)))
            if not chunk:
                continue
            if chunk.get("section_path", "") != sec:
                continue
            text = chunk.get("text", "")
            if not text:
                continue
            token_len = len(enc.encode(text))
            if total_tokens + token_len > max_tokens:
                # Truncate the remaining space
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 0:
                    truncated = enc.decode(enc.encode(text)[:remaining_tokens])
                    collected.append((chunk.get("chunk_id", f"{src}:{nb}"), truncated))
                return collected
            collected.append((chunk.get("chunk_id", f"{src}:{nb}"), text))
            total_tokens += token_len
            seen.add(key)
    return collected


def build_prompt(query: str, context_chunks: List[str]) -> List[dict]:
    # Kept for backward compatibility; delegate to shared prompt builder
    return build_messages(query, context_chunks, language="English")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid dense+sparse retrieval with BGE rerank")
    parser.add_argument("--query", required=True, help="user query text")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/markdown/chunked/chunks.jsonl"),
        help="path to chunked JSONL for BM25",
    )
    parser.add_argument(
        "--persist-path",
        type=Path,
        default=Path("chroma_db"),
        help="Chroma persistence dir",
    )
    parser.add_argument(
        "--collection",
        default="coffee_text",
        help="Chroma collection name for dense",
    )
    parser.add_argument("--k-dense", type=int, default=40)
    parser.add_argument("--k-sparse", type=int, default=40)
    parser.add_argument("--top-fuse", type=int, default=40)
    parser.add_argument("--top-rerank", type=int, default=6)
    parser.add_argument("--neighbor-radius", type=int, default=2, help="How many neighbor blocks to include around each hit")
    parser.add_argument("--max-context-tokens", type=int, default=3000, help="Max tokens to pass to LLM")
    parser.add_argument("--encoding", default="cl100k_base", help="Tokenizer encoding name for token counting")
    parser.add_argument(
        "--dense-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model used in collection",
    )
    parser.add_argument(
        "--rerank-model",
        default="BAAI/bge-reranker-base",
        help="BGE reranker checkpoint",
    )
    parser.add_argument(
        "--chat-model",
        default="gpt-4o-mini",
        help="Chat model name for final answer (OpenAI-compatible)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM answer and only print retrieved context",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    # Dense embedder
    embedder = OpenAIEmbeddings(model=args.dense_model)

    # Chroma client
    client = chromadb.PersistentClient(path=str(args.persist_path))
    collection = client.get_collection(args.collection)

    # Sparse BM25
    chunks = load_chunks(args.chunks_path)
    bm25 = build_bm25(chunks)
    chunk_map = load_chunk_map(chunks)

    # Dense search
    q_vec = embedder.embed_query(args.query)
    dense_hits = dense_search(collection, q_vec, args.k_dense)

    # Sparse search
    sparse_hits = sparse_search(bm25, args.query, args.k_sparse)

    # RRF fusion
    fused = rrf_fuse(dense_hits, sparse_hits, top_n=args.top_fuse)

    # Rerank
    reranker = FlagReranker(args.rerank_model, use_fp16=True)
    reranked = rerank_bge(reranker, args.query, fused, top_n=args.top_rerank * 2)
    reranked = dedup_results(reranked)[: args.top_rerank]

    # Collect context with neighbors
    collected = collect_with_neighbors(
        reranked,
        chunk_map,
        radius=args.neighbor_radius,
        max_tokens=args.max_context_tokens,
        encoder_name=args.encoding,
    )
    # Fallback: if no context collected, use reranked docs directly (trimmed)
    if not collected:
        enc = get_encoding(args.encoding)
        total = 0
        for _id, _sc, meta, doc in reranked:
            tokens = enc.encode(doc)
            if total >= args.max_context_tokens:
                break
            remain = args.max_context_tokens - total
            part = enc.decode(tokens[:remain])
            cid = meta.get("chunk_id", f"{meta.get('source','')}:{meta.get('block_idx',-1)}")
            collected.append((cid, part))
            total += len(tokens[:remain])

    chunk_ids = [cid for cid, _ in collected]
    context_chunks = [txt for _, txt in collected]

    def safe_print(text: str):
        """Print text safely, replacing chars that console encoding cannot handle."""
        import sys
        try:
            print(text)
        except UnicodeEncodeError:
            # Get console encoding (e.g., 'gbk' on Windows, 'utf-8' on Linux)
            console_encoding = sys.stdout.encoding or "utf-8"
            # Encode to console encoding, replacing problematic chars with '?'
            safe_bytes = text.encode(console_encoding, errors="replace")
            safe_text = safe_bytes.decode(console_encoding)
            print(safe_text)

    safe_print(f"\nQuery: {args.query}")
    safe_print(f"Dense hits: {len(dense_hits)}, Sparse hits: {len(sparse_hits)}, Fused: {len(fused)}, Reranked: {len(reranked)}")
    for i, (_id, sc, meta, doc) in enumerate(reranked, start=1):
        src = meta.get("source", "")
        sec = meta.get("section_path", "")
        safe_print(f"\n#{i} score={sc:.4f} source={src} section={sec}")
        safe_print(doc[:400].replace("\n", " "))

    safe_print("\nContext for LLM (neighbor-augmented, token-limited):")
    for i, (cid, chunk) in enumerate(zip(chunk_ids, context_chunks), start=1):
        safe_print(f"\n[CTX {i}] chunk_id={cid} {chunk[:400].replace(chr(10), ' ')}")

    if not args.no_llm:
        # First-pass answer
        top1 = reranked[0][1] if reranked else 0.0
        avg_top5 = sum(r[1] for r in reranked[:5]) / max(1, min(5, len(reranked)))
        low_retrieval_conf = (top1 < 0.35) or (avg_top5 < 0.30)
        messages = build_messages_answer(
            args.query, context_chunks, chunk_ids, low_retrieval_conf=low_retrieval_conf, language="English"
        )
        resp = generate_answer(messages)
        answer_text = resp.content
        safe_print("\nLLM Answer (raw):\n")
        safe_print(answer_text)

        # Judge pass
        judge_messages = build_messages_judge(
            args.query, context_chunks, chunk_ids, answer_text, language="English"
        )
        judge_resp = generate_answer(judge_messages)
        safe_print("\nJudge:\n")
        safe_print(judge_resp.content)
        
        # Parse LLM answer and Judge, then output final JSON
        try:
            # Try to parse LLM answer as JSON
            answer_json = json.loads(answer_text)
            can_answer = answer_json.get("can_answer", True)
            confidence = answer_json.get("confidence", 0.5)
            answer = answer_json.get("answer", answer_text)
            reason = answer_json.get("reason", "")
            sources = answer_json.get("sources", chunk_ids)
        except json.JSONDecodeError:
            # Fallback: treat as plain text answer
            can_answer = True
            confidence = 0.7
            answer = answer_text
            reason = "Plain text answer"
            sources = chunk_ids
        
        # Try to parse Judge
        try:
            judge_json = json.loads(judge_resp.content)
            is_supported = judge_json.get("is_supported", True)
            hallucination_level = judge_json.get("hallucination_level", 0)
            judge_confidence = judge_json.get("overall_confidence", confidence)
            
            # If Judge says not supported or high hallucination, override
            if not is_supported or hallucination_level >= 1:
                can_answer = False
                confidence = min(confidence, judge_confidence)
                reason = judge_json.get("comment", "Judge detected issues")
        except json.JSONDecodeError:
            pass  # Keep original answer
        
        # Output final JSON result for rag_tool to parse
        final_result = {
            "can_answer": can_answer,
            "confidence": confidence,
            "answer": answer,
            "reason": reason,
            "sources": sources,
            "retrieval_scores": {
                "top1": top1,
                "avg_top5": avg_top5,
                "hits": len(reranked)
            }
        }
        safe_print("\n" + "="*80)
        safe_print("FINAL_JSON_RESULT:")
        safe_print(json.dumps(final_result, ensure_ascii=False))
        safe_print("="*80)


if __name__ == "__main__":
    main()

