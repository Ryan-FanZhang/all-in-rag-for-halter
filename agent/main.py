"""Simple agent entry: router -> RAG (hybrid_retrieve) or escalate."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Allow importing utils.*
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.router_chain import parse_signals, build_router_messages  


def decide_action(query: str, signals: dict, model: str, temperature: float):
    top1 = float(signals.get("top1", 0) or 0)
    avg5 = float(signals.get("avg_top5", 0) or 0)
    hits = int(signals.get("hits", 0) or 0)
    if top1 >= 0.7 and avg5 >= 0.5 and hits >= 3:
        return {"action": "rag", "confidence": 0.9, "reason": "High retrieval scores"}
    if top1 < 0.35 or avg5 < 0.30 or hits < 3:
        return {"action": "escalate", "confidence": 0.9, "reason": "Low retrieval scores"}

    messages = build_router_messages(query, signals)
    llm = ChatOpenAI(model=model, temperature=temperature)
    resp = llm.invoke(messages)
    try:
        return json.loads(resp.content)
    except Exception:
        return {"action": "escalate", "confidence": 0.5, "reason": "Router parse failure"}


def run_hybrid_retrieve(query: str):
    cmd = [
        sys.executable,
        str(ROOT / "utils" / "hybrid_retrieve.py"),
        "--query",
        query,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Agent: router -> rag|escalate")
    parser.add_argument("--query", required=True, help="user question")
    parser.add_argument(
        "--retrieval-signals",
        default="{}",
        help='JSON string, e.g. {"top1":0.3,"avg_top5":0.25,"hits":5,"sections":["TROUBLESHOOTING"]}',
    )
    parser.add_argument("--top1", type=float, default=None)
    parser.add_argument("--avg-top5", dest="avg_top5", type=float, default=None)
    parser.add_argument("--hits", type=int, default=None)
    parser.add_argument("--sections", nargs="*", default=None)
    parser.add_argument("--router-model", default="gpt-4o-mini")
    parser.add_argument("--router-temperature", type=float, default=0.0)
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

    decision = decide_action(args.query, signals, args.router_model, args.router_temperature)
    print(json.dumps({"router_decision": decision, "signals": signals}, ensure_ascii=False))

    if decision.get("action") == "rag":
        out, err, rc = run_hybrid_retrieve(args.query)
        print("=== RAG OUTPUT ===")
        print(out)
        if err:
            print("=== RAG STDERR ===")
            print(err)
    else:
        print("=== ESCALATE ===")
        print("Escalating to support/help desk based on routing decision.")


if __name__ == "__main__":
    main()

