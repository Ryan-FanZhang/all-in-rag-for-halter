"""RAG Tool: Retrieval-Augmented Generation for answering questions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from langchain.tools import BaseTool
from pydantic import Field

# Ensure project root import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class RAGTool(BaseTool):
    """Tool for answering questions using RAG (Retrieval-Augmented Generation).
    
    This tool retrieves relevant context from the knowledge base and generates
    an answer using an LLM with confidence scoring and source citation.
    """

    name: str = "rag_search"
    description: str = (
        "Use this tool to answer questions about coffee machine troubleshooting, "
        "product features, maintenance, and user manual content. "
        "Input should be a clear question string."
    )
    
    # Tool configuration (matching hybrid_retrieve.py defaults)
    k_dense: int = Field(default=40, description="Number of dense retrieval results")
    k_sparse: int = Field(default=40, description="Number of sparse (BM25) retrieval results")
    top_fuse: int = Field(default=40, description="Number of results after RRF fusion")
    top_rerank: int = Field(default=6, description="Number of top reranked results to use")
    neighbor_radius: int = Field(default=2, description="Number of neighbor chunks to expand around each hit")
    max_context_tokens: int = Field(default=3000, description="Maximum context tokens for LLM")
    encoding: str = Field(default="cl100k_base", description="Tokenizer encoding name")
    dense_model: str = Field(default="text-embedding-3-large", description="OpenAI embedding model")
    rerank_model: str = Field(default="BAAI/bge-reranker-base", description="BGE reranker model")
    chat_model: str = Field(default="gpt-5", description="Chat model for answer generation")       

    def _run(self, query: str) -> str:
        """Execute RAG retrieval and answer generation."""
        try:
            # Call hybrid_retrieve.py as subprocess with all parameters
            rag_args = [
                "--query", query,
                "--k-dense", str(self.k_dense),
                "--k-sparse", str(self.k_sparse),
                "--top-fuse", str(self.top_fuse),
                "--top-rerank", str(self.top_rerank),
                "--neighbor-radius", str(self.neighbor_radius),
                "--max-context-tokens", str(self.max_context_tokens),
                "--encoding", self.encoding,
                "--dense-model", self.dense_model,
                "--rerank-model", self.rerank_model,
                "--chat-model", self.chat_model,
            ]
            
            rag_process = subprocess.run(
                [sys.executable, str(ROOT / "utils" / "hybrid_retrieve.py")] + rag_args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=120,  # 120 second timeout (increased for reranker loading)
            )
            
            if rag_process.returncode != 0:
                error_msg = f"RAG process failed: {rag_process.stderr[:500]}"
                return json.dumps({
                    "status": "error",
                    "message": error_msg,
                    "can_answer": False,
                    "confidence": 0.0
                }, ensure_ascii=False)
            
            # Parse the output to extract the JSON answer
            stdout = rag_process.stdout.strip()
            
            # Look for special marker FINAL_JSON_RESULT
            if "FINAL_JSON_RESULT:" in stdout:
                parts = stdout.split("FINAL_JSON_RESULT:")
                if len(parts) > 1:
                    json_part = parts[-1].split("="*80)[0].strip()
                    try:
                        result = json.loads(json_part)
                        return json.dumps(result, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass
            
            # Fallback: Try to find JSON in the output (it should be at the end)
            lines = stdout.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        result = json.loads(line)
                        if "can_answer" in result or "answer" in result:
                            return json.dumps(result, ensure_ascii=False)
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, return the full output
            return json.dumps({
                "status": "success",
                "message": stdout[-1000:],  # Last 1000 chars
                "raw_output": True
            }, ensure_ascii=False)
            
        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "error",
                "message": "RAG process timed out (>60s)",
                "can_answer": False,
                "confidence": 0.0
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "can_answer": False,
                "confidence": 0.0
            }, ensure_ascii=False)

    async def _arun(self, query: str) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(query)


def create_rag_tool(
    k_dense: int = 40,
    k_sparse: int = 40,
    top_fuse: int = 40,
    top_rerank: int = 6,
    neighbor_radius: int = 2,
    max_context_tokens: int = 3000,
    encoding: str = "cl100k_base",
    dense_model: str = "text-embedding-3-large",
    rerank_model: str = "BAAI/bge-reranker-base",
    chat_model: str = "gpt-4o-mini",
) -> RAGTool:
    """Factory function to create a configured RAG tool.
    
    All default values match hybrid_retrieve.py for consistency.
    """
    return RAGTool(
        k_dense=k_dense,
        k_sparse=k_sparse,
        top_fuse=top_fuse,
        top_rerank=top_rerank,
        neighbor_radius=neighbor_radius,
        max_context_tokens=max_context_tokens,
        encoding=encoding,
        dense_model=dense_model,
        rerank_model=rerank_model,
        chat_model=chat_model,
    )

