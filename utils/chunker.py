from pathlib import Path
import json
import sys
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 默认源文件路径可被命令行参数覆盖
default_src = Path(
    "data/markdown/MinerU_markdown_3d9_EN-5713226261_20251210044802_1998494870494453760.md"
)
src = Path(sys.argv[1]) if len(sys.argv) > 1 else default_src
if not src.exists():
    raise FileNotFoundError(f"源文件不存在: {src}")

text = src.read_text(encoding="utf-8")

# 先按标题切分，提取层级元数据
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
sections = header_splitter.split_text(text)

# 再做细分：512 tokens 等价的字符长度近似，重叠 200
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

docs = []
for sec in sections:
    section_path = " / ".join(
        [
            sec.metadata.get("h1", "") or "",
            sec.metadata.get("h2", "") or "",
            sec.metadata.get("h3", "") or "",
        ]
    ).strip(" /")
    for i, chunk in enumerate(splitter.split_text(sec.page_content)):
        docs.append(
            {
                "text": chunk,
                "section_path": section_path,
                "source": src.name,
                "block_idx": i,
                "block_type": sec.metadata.get("block_type", "text"),
            }
        )

out = Path("data/markdown/chunked/chunks.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"wrote {len(docs)} chunks -> {out}")