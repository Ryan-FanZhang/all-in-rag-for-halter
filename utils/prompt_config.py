from __future__ import annotations

from typing import List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage


def build_messages_answer(
    query: str,
    context_chunks: Sequence[str],
    chunk_ids: Sequence[str],
    low_retrieval_conf: bool,
    language: str = "Chinese",
) -> List[BaseMessage]:
    system_text = (
        "You are a manual assistant with strong experience in coffee machine maintenance and troubleshooting. "
        "Only use the provided context; do not fabricate. "
        "If context is insufficient, say you cannot find reliable info and tell customers to contact support. "
        "Always cite chunk_id and section/source. Provide a confidence score (0.0-1.0) reflecting how well the context supports the answer. "
        "If confidence < 0.6 OR low_retrieval_conf=true, set can_answer=false and suggest contacting support."
    )
    ctx_lines = []
    for i, (cid, c) in enumerate(zip(chunk_ids, context_chunks), start=1):
        ctx_lines.append(f"[CTX {i}] chunk_id={cid}\n{c}")
    ctx = "\n\n".join(ctx_lines)
    user_text = (
        f"low_retrieval_conf: {str(low_retrieval_conf).lower()}\n"
        f"Question: {query}\n"
        f"Context:\n{ctx}\n"
        f"Respond ONLY in JSON, no extra text:\n"
        "{\n"
        '  "can_answer": true/false,\n'
        '  "confidence": number 0.0-1.0,\n'
        '  "answer": "text answer in ' + language + '",\n'
        '  "reason": "brief reason",\n'
        '  "sources": ["chunk_id1", "chunk_id2"]\n'
        "}\n"
        "Rules: If can_answer=false OR confidence<0.5 OR low_retrieval_conf=true, set can_answer=false, confidence=0.0, answer='No reliable information found, please contact support', reason explain insufficiency, sources=[]"
    )
    return [SystemMessage(content=system_text), HumanMessage(content=user_text)]


def build_messages_judge(
    query: str,
    context_chunks: Sequence[str],
    chunk_ids: Sequence[str],
    answer_text: str,
    language: str = "Chinese",
) -> List[BaseMessage]:
    system_text = (
        "You are a strict judge. Evaluate if the assistant answer is supported by context. "
        "Only judge support; do not invent new info."
    )
    ctx_lines = []
    for i, (cid, c) in enumerate(zip(chunk_ids, context_chunks), start=1):
        ctx_lines.append(f"[CTX {i}] chunk_id={cid}\n{c}")
    ctx = "\n\n".join(ctx_lines)
    user_text = (
        f"Question: {query}\n"
        f"Context:\n{ctx}\n"
        f"Assistant answer:\n{answer_text}\n"
        "Respond ONLY in JSON:\n"
        "{\n"
        '  "is_supported": true/false,\n'
        '  "hallucination_level": 0/1/2,\n'
        '  "missing_info": true/false,\n'
        '  "overall_confidence": 0.0-1.0,\n'
        '  "comment": "brief reason"\n'
        "}\n"
        "If not supported OR hallucination_level>=1, set overall_confidence<=0.5."
    )
    return [SystemMessage(content=system_text), HumanMessage(content=user_text)]

