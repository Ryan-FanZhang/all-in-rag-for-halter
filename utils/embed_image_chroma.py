"""Embed images from markdown into Chroma using OpenCLIP ViT-g-14."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import chromadb
import requests
import torch
from dotenv import load_dotenv
from langchain_experimental.open_clip import OpenCLIPEmbeddings


def parse_markdown_images(md_path: Path) -> List[dict]:
    """Extract image entries with section path and alt text."""
    img_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
    header_pattern = re.compile(r"^(#{1,6})\s*(.+)")

    section = {"h1": "", "h2": "", "h3": ""}
    results: List[dict] = []

    with md_path.open(encoding="utf-8") as f:
        for line in f:
            h = header_pattern.match(line)
            if h:
                level = len(h.group(1))
                title = h.group(2).strip()
                if level == 1:
                    section["h1"], section["h2"], section["h3"] = title, "", ""
                elif level == 2:
                    section["h2"], section["h3"] = title, ""
                elif level == 3:
                    section["h3"] = title

            for m in img_pattern.finditer(line):
                alt, url = m.groups()
                section_path = " / ".join(
                    [section["h1"], section["h2"], section["h3"]]
                ).strip(" /")
                results.append(
                    {
                        "url": url.strip(),
                        "alt": alt.strip(),
                        "section_path": section_path,
                        "source": md_path.name,
                    }
                )
    return results


def download_image(url: str, dest_dir: Path, idx: int) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix
    if not ext:
        ext = ".jpg"
    local_path = dest_dir / f"img_{idx}{ext}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        return local_path
    except Exception as e:
        print(f"failed to download {url}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed markdown images into Chroma")
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=Path(
            "data/markdown/MinerU_markdown_3d9_EN-5713226261_20251210044802_1998494870494453760.md"
        ),
        help="Path to source markdown file",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory to save downloaded images",
    )
    parser.add_argument(
        "--persist-path",
        type=Path,
        default=Path("chroma_db"),
        help="Chroma persistence directory",
    )
    parser.add_argument(
        "--collection",
        default="coffee_image",
        help="Chroma collection name for images",
    )
    parser.add_argument(
        "--delete-source",
        default=None,
        help="If set, delete existing entries with this source before upsert",
    )
    args = parser.parse_args()

    load_dotenv()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    entries = parse_markdown_images(args.markdown_path)
    print(f"found {len(entries)} images in markdown")
    if not entries:
        return

    downloaded: List[Tuple[dict, Path]] = []
    for idx, entry in enumerate(entries):
        local = download_image(entry["url"], args.images_dir, idx)
        if local:
            downloaded.append((entry, local))

    print(f"downloaded {len(downloaded)} images")
    if not downloaded:
        return

    embedder = OpenCLIPEmbeddings(
        model_name="ViT-B-32",
        checkpoint="laion2b_s34b_b79k",
        device=device,
    )

    chroma_client = chromadb.PersistentClient(path=str(args.persist_path))
    collection = chroma_client.get_or_create_collection(args.collection)
    if args.delete_source:
        collection.delete(where={"source": args.delete_source})
        print(f"deleted existing images with source={args.delete_source}")

    valid_ids = []
    valid_vecs = []
    valid_metas = []
    for i, (entry, path) in enumerate(downloaded):
        try:
            vec = embedder.embed_image([str(path)])[0]
        except Exception as e:
            print(f"failed to embed {path}: {e}")
            continue

        valid_ids.append(f"img-{i}")
        valid_vecs.append(vec)
        valid_metas.append(
            {
                "section_path": entry.get("section_path", ""),
                "source": entry.get("source", ""),
                "url": entry.get("url", ""),
                "alt": entry.get("alt", ""),
                "local_path": str(path),
                "block_type": "image",
            }
        )

    if not valid_ids:
        print("no valid embeddings to upsert")
        return

    collection.upsert(ids=valid_ids, embeddings=valid_vecs, metadatas=valid_metas)
    print(
        f"upserted {len(valid_ids)} image vectors -> collection={args.collection}, size={collection.count()}"
    )


if __name__ == "__main__":
    main()

