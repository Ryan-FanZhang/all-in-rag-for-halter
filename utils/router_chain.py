"""Simple LLM router to choose action: rag / escalate / db / api."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Ensure project root import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def parse_signals(raw: str) -> dict:
    """Robustly parse retrieval signals from CLI string."""
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s:
        return {}
    candidates = [s]
    # Strip outer quotes if present
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        candidates.append(s[1:-1])
    # Try replacing single quotes with double quotes
    candidates.append(s.replace("'", '"'))

    # Try fixing missing quotes around keys/array items
    import re

    def fix_lists(txt: str) -> str:
        def repl(m):
            items = [p.strip() for p in m.group(1).split(",") if p.strip()]
            fixed_items = []
            for p in items:
                if p.startswith('"') or p.startswith("'"):
                    val = p.strip('"').strip("'")
                    fixed_items.append(f'"{val}"')
                elif re.fullmatch(r"-?\d+(\.\d+)?", p):
                    fixed_items.append(p)
                else:
                    fixed_items.append(f'"{p}"')
            return "[" + ",".join(fixed_items) + "]"

        return re.sub(r"\[(.*?)\]", repl, txt)

    fixed = re.sub(r"(\b\w+\b)\s*:", r'"\1":', s)
    fixed = fix_lists(fixed)
    candidates.append(fixed)

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    try:
        import ast

        return ast.literal_eval(s)
    except Exception:
        return {}


def build_router_messages(query: str, retrieval_signals: dict) -> List[dict]:
    system_text = (
        "You are a routing assistant for a coffee machine support system. Decide the best action for the user query.\n"
        "Possible actions: rag, escalate, db, api.\n\n"
        
        "## Hard Rules (ALWAYS apply these first):\n"
        "1. If top1>=0.5 AND avg_top5>=0.35 AND hits>=3 → action=rag, confidence=0.9\n"
        "2. If top1<0.15 OR avg_top5<0.10 OR hits<2 → action=escalate, confidence=0.9\n"
        "3. Otherwise, use intent-based routing below (PRIORITIZE API for order/inventory/price queries)\n\n"
        
        "## Intent-Based Routing (if hard rules don't apply):\n\n"
        
        "### API (confidence=0.85) - Use when query asks for LIVE/REAL-TIME data:\n"
        "**Trigger keywords**: order, status, tracking, inventory, stock, availability, price, shipping, delivery, service status, uptime, payment, transaction\n"
        "**Examples**:\n"
        "- 'check my order ORD12345' → api\n"
        "- 'what is my order status' → api\n"
        "- 'is the coffee machine in stock' → api\n"
        "- 'check inventory' → api\n"
        "- 'how much does it cost' → api (real-time price)\n"
        "- 'what is the service status' → api\n"
        "- 'track my shipment' → api\n\n"
        
        "### RAG (confidence=0.75) - Use for STATIC knowledge from manuals:\n"
        "**Trigger keywords**: how to, troubleshoot, fix, problem, issue, manual, guide, instructions, features, specifications\n"
        "**Examples**:\n"
        "- 'my coffee is not hot' → rag (troubleshooting)\n"
        "- 'how to clean the machine' → rag (instructions)\n"
        "- 'what does the red light mean' → rag (manual)\n"
        "- 'how to adjust grinder' → rag (how-to)\n\n"
        
        "### ESCALATE (confidence=0.85) - Use when cannot be handled:\n"
        "**Trigger keywords**: complaint, refund, warranty claim, legal, safety concern, human agent, speak to someone\n"
        "**Examples**:\n"
        "- 'I want a refund' → escalate\n"
        "- 'this is unacceptable' → escalate\n"
        "- 'connect me to support' → escalate\n\n"
        
        "### DB (confidence=0.80) - Use for structured data lookups:\n"
        "**Examples**: customer records, purchase history, warranty records (not implemented yet)\n\n"
        
        "## Decision Priority:\n"
        "1. Check hard rules first (retrieval scores)\n"
        "2. **HIGHEST PRIORITY**: If query contains API keywords (order/status/tracking/inventory/stock/price) → api\n"
        "3. If query is about product usage/troubleshooting → rag\n"
        "4. If query needs human intervention → escalate\n\n"
        "**CRITICAL**: Queries about orders, inventory, prices, or service status should ALWAYS use 'api', "
        "even if retrieval scores are low. Low retrieval scores for such queries are EXPECTED because this "
        "information is not in the knowledge base.\n\n"
        
        "Respond ONLY in JSON: {\"action\": \"rag|escalate|db|api\", \"confidence\": 0.0-1.0, \"reason\": \"...\"}"
    )
    user_text = (
        f"query: {query}\n"
        f"retrieval_signals: {json.dumps(retrieval_signals, ensure_ascii=False)}\n"
        "Apply the numeric rules first. If a rule matches, output it directly. Otherwise, decide by intent."
    )
    return [SystemMessage(content=system_text), HumanMessage(content=user_text)]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Router")
    parser.add_argument("--query", required=True, help="user query")
    parser.add_argument(
        "--retrieval-signals",
        default="{}",
        help='JSON string, e.g. {"top1":0.3,"avg_top5":0.25,"hits":5,"sections":["TROUBLESHOOTING"]}',
    )
    parser.add_argument("--top1", type=float, default=None, help="override top1 similarity")
    parser.add_argument("--avg-top5", dest="avg_top5", type=float, default=None, help="override avg_top5 similarity")
    parser.add_argument("--hits", type=int, default=None, help="override hits count")
    parser.add_argument("--sections", nargs="*", default=None, help="override sections list")
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="OpenAI-compatible chat model for routing"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    load_dotenv()
    signals = parse_signals(args.retrieval_signals)
    if args.top1 is not None:
        signals["top1"] = args.top1
    if args.avg_top5 is not None:
        signals["avg_top5"] = args.avg_top5
    if args.hits is not None:
        signals["hits"] = args.hits
    if args.sections is not None:
        signals["sections"] = args.sections

    # Hard rule override
    top1 = float(signals.get("top1", 0) or 0)
    avg5 = float(signals.get("avg_top5", 0) or 0)
    hits = int(signals.get("hits", 0) or 0)
    debug_info = {
        "top1": top1,
        "avg_top5": avg5,
        "hits": hits,
        "raw_signals": signals,
        "raw_input": args.retrieval_signals,
    }
    print(json.dumps({"debug_signals": debug_info}, ensure_ascii=False))

    # Adjusted rules: more lenient for RAG, stricter for escalate
    if top1 >= 0.5 and avg5 >= 0.35 and hits >= 3:
        result = {"action": "rag", "confidence": 0.9, "reason": "High retrieval scores (relaxed threshold)"}
        print(json.dumps(result, ensure_ascii=False))
        return
    
    # IMPORTANT: Only escalate on extremely low scores
    # For API queries (order/inventory/price), let LLM decide even if scores are low
    if top1 < 0.15 or avg5 < 0.10 or hits < 2:
        result = {"action": "escalate", "confidence": 0.9, "reason": "Extremely low retrieval scores"}
        print(json.dumps(result, ensure_ascii=False))
        return

    # Otherwise, defer to LLM for intent-based routing
    messages = build_router_messages(args.query, signals)
    llm = ChatOpenAI(model=args.model, temperature=args.temperature)
    resp = llm.invoke(messages)
    print(resp.content)


if __name__ == "__main__":
    main()

