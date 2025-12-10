"""Embed markdown chunks into Chroma using OpenAI text-embedding-3-large."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator, List

import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


def batched(items: List[dict], batch_size: int) -> Iterator[List[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_chunks(path: Path) -> List[dict]:
    data: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed markdown chunks into Chroma")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/markdown/chunked/chunks.jsonl"),
        help="Path to chunked JSONL file",
    )
    parser.add_argument(
        "--persist-path",
        type=Path,
        default=Path("chroma_db"),
        help="Directory for Chroma persistence",
    )
    parser.add_argument(
        "--collection",
        default="coffee_text",
        help="Chroma collection name for text embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding requests",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
        help="OpenAI embedding model",
    )
    parser.add_argument(
        "--delete-source",
        default=None,
        help="If set, delete existing entries with this source before upsert",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY")

    chunks = load_chunks(args.chunks_path)
    print(f"loaded {len(chunks)} chunks from {args.chunks_path}")

    chroma_client = chromadb.PersistentClient(path=str(args.persist_path))
    collection = chroma_client.get_or_create_collection(args.collection)

    if args.delete_source:
        collection.delete(where={"source": args.delete_source})
        print(f"deleted existing docs with source={args.delete_source}")

    # LangChain OpenAI embeddings (env var OPENAI_API_KEY / AZURE settings)
    embeddings = OpenAIEmbeddings(model=args.model)

    total = 0
    for batch in batched(chunks, args.batch_size):
        texts = [d["text"] for d in batch]
        ids = [f'text-{d.get("block_idx", i)}-{i}' for i, d in enumerate(batch, start=total)]
        metas = [
            {
                "section_path": d.get("section_path", ""),
                "source": d.get("source", ""),
                "block_idx": d.get("block_idx", -1),
                "block_type": d.get("block_type", "text"),
            }
            for d in batch
        ]

        vectors = embeddings.embed_documents(texts)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metas,
        )
        total += len(batch)
        print(f"upserted {total}/{len(chunks)}")

    print(
        f"done. collection={args.collection}, size={collection.count()}, persisted at {args.persist_path}"
    )


if __name__ == "__main__":
    main()

