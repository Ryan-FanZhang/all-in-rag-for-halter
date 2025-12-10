"""Streamlit frontend for RAG chatbot with routing."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Ensure project root import
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.rag_tool import create_rag_tool
from tools.escalate_tool import create_escalate_tool

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Assistant Support",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - Dark Theme
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #1a1a1a;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Base chat message styling - black bubble with white text */
    .chat-message {
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        background-color: #000000;
        color: #ffffff;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .chat-message strong {
        color: #ffffff;
    }
    
    /* User message - black with blue accent */
    .user-message {
        background-color: #000000;
        border-left: 5px solid #2196f3;
        color: #ffffff;
    }
    
    /* Assistant message - black with green accent */
    .assistant-message {
        background-color: #000000;
        border-left: 5px solid #4caf50;
        color: #ffffff;
    }
    
    /* Error message - black with red accent */
    .error-message {
        background-color: #000000;
        border-left: 5px solid #f44336;
        color: #ffffff;
    }
    
    /* Escalate message - black with orange accent */
    .escalate-message {
        background-color: #000000;
        border-left: 5px solid #ff9800;
        color: #ffffff;
    }
    
    /* Confidence indicators - keep original colors for visibility */
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    
    /* Ensure all text in chat messages is white */
    .chat-message p, .chat-message div, .chat-message span {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def call_router(query: str, retrieval_signals: Optional[dict] = None) -> dict:
    """Call the router to decide which action to take."""
    if retrieval_signals is None:
        retrieval_signals = {}
    
    try:
        router_args = [
            "--query", query,
            "--retrieval-signals", json.dumps(retrieval_signals, ensure_ascii=False)
        ]
        
        router_process = subprocess.run(
            [sys.executable, str(ROOT / "utils" / "router_chain.py")] + router_args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        
        if router_process.returncode != 0:
            return {
                "action": "escalate",
                "confidence": 0.0,
                "reason": "Router error",
                "error": router_process.stderr[:500]
            }
        
        # Parse output - router prints debug info and then the decision
        lines = router_process.stdout.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('{') and '"action"' in line:
                try:
                    result = json.loads(line)
                    if "action" in result:
                        return result
                except json.JSONDecodeError:
                    continue
        
        # Fallback
        return {
            "action": "escalate",
            "confidence": 0.5,
            "reason": "Could not parse router output"
        }
        
    except subprocess.TimeoutExpired:
        return {
            "action": "escalate",
            "confidence": 0.0,
            "reason": "Router timeout"
        }
    except Exception as e:
        return {
            "action": "escalate",
            "confidence": 0.0,
            "reason": f"Router exception: {str(e)}"
        }


def format_confidence(confidence: float) -> str:
    """Format confidence score with color coding."""
    if confidence >= 0.7:
        cls = "confidence-high"
        label = "High"
    elif confidence >= 0.4:
        cls = "confidence-medium"
        label = "Medium"
    else:
        cls = "confidence-low"
        label = "Low"
    return f'<span class="{cls}">{label} ({confidence:.2f})</span>'


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">☕ Coffee Machine Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("RAG Configuration")
        k_dense = st.slider("Dense Retrieval (K)", 10, 100, 40, help="Number of dense retrieval results")
        k_sparse = st.slider("Sparse Retrieval (K)", 10, 100, 40, help="Number of BM25 sparse retrieval results")
        top_fuse = st.slider("Top Fusion Results", 10, 100, 40, help="Number of results after RRF fusion")
        top_rerank = st.slider("Top Rerank Results", 1, 20, 6, help="Number of top documents to use after reranking")
        neighbor_radius = st.slider("Neighbor Radius", 0, 5, 2, help="Number of neighboring chunks to include around each hit")
        max_context_tokens = st.slider("Max Context Tokens", 500, 5000, 3000, step=100, help="Maximum tokens for LLM context")
        
        st.markdown("---")
        st.subheader("Router Override")
        use_manual_signals = st.checkbox("Manual Retrieval Signals", value=False, help="Override router with manual signals")
        
        manual_signals = {}
        if use_manual_signals:
            manual_signals["top1"] = st.slider("Top1 Score", 0.0, 1.0, 0.5, 0.01)
            manual_signals["avg_top5"] = st.slider("Avg Top5 Score", 0.0, 1.0, 0.4, 0.01)
            manual_signals["hits"] = st.number_input("Hits Count", 0, 100, 5, 1)
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        metadata = message.get("metadata", {})
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{content}</div>', unsafe_allow_html=True)
        elif role == "assistant":
            action = metadata.get("action", "unknown")
            confidence = metadata.get("confidence", 0.0)
            
            if action == "escalate":
                msg_class = "escalate-message"
                icon = "🎫"
            elif metadata.get("error"):
                msg_class = "error-message"
                icon = "❌"
            else:
                msg_class = "assistant-message"
                icon = "✅"
            
            st.markdown(
                f'<div class="chat-message {msg_class}">'
                f'<strong>{icon} Assistant (Action: {action}, Confidence: {format_confidence(confidence)}):</strong><br>'
                f'{content}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            # Show sources if available
            if metadata.get("sources"):
                with st.expander("📚 View Sources"):
                    st.json(metadata["sources"])
    
    # Chat input
    user_query = st.chat_input("Ask me anything about your coffee machine...")
    
    if user_query:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Show user message
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong><br>{user_query}</div>', unsafe_allow_html=True)
        
        # Processing indicator
        with st.spinner("🤔 Thinking..."):
            # Step 1: Pre-retrieve to get real retrieval signals (unless manual override)
            if not use_manual_signals:
                with st.spinner("🔍 Analyzing query..."):
                    # Quick retrieval to get signals for router
                    try:
                        import chromadb
                        from langchain_openai import OpenAIEmbeddings
                        from dotenv import load_dotenv
                        import os
                        
                        load_dotenv()
                        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
                        client = chromadb.PersistentClient(path="chroma_db")
                        collection = client.get_collection("coffee_text")
                        
                        # Get query embedding and search
                        q_vec = embedder.embed_query(user_query)
                        res = collection.query(
                            query_embeddings=[q_vec],
                            n_results=5,
                            include=["metadatas", "distances"],
                        )
                        
                        # Calculate retrieval signals
                        if res["ids"][0]:
                            distances = res["distances"][0]
                            scores = [1.0 / (1.0 + d) for d in distances]
                            top1 = scores[0] if scores else 0.0
                            avg_top5 = sum(scores[:5]) / len(scores) if scores else 0.0
                            hits = len(scores)
                            sections = list(set([m.get("section_path", "") for m in res["metadatas"][0] if m.get("section_path")]))
                            
                            retrieval_signals = {
                                "top1": top1,
                                "avg_top5": avg_top5,
                                "hits": hits,
                                "sections": sections[:3]  # Top 3 sections
                            }
                        else:
                            retrieval_signals = {"top1": 0.0, "avg_top5": 0.0, "hits": 0, "sections": []}
                    except Exception as e:
                        st.warning(f"Pre-retrieval failed: {str(e)[:100]}. Using empty signals.")
                        retrieval_signals = {"top1": 0.0, "avg_top5": 0.0, "hits": 0, "sections": []}
            else:
                retrieval_signals = manual_signals
            
            # Show retrieval signals in sidebar
            with st.sidebar:
                st.success(f"**🔍 Retrieval Signals:**\n\n"
                          f"- Top1: **{retrieval_signals.get('top1', 0):.4f}**\n"
                          f"- Avg Top5: **{retrieval_signals.get('avg_top5', 0):.4f}**\n"
                          f"- Hits: **{retrieval_signals.get('hits', 0)}**\n"
                          f"- Sections: {', '.join(retrieval_signals.get('sections', [])[:2]) or 'None'}\n\n"
                          f"*✅ RAG if: top1≥0.5 AND avg5≥0.35 AND hits≥3*\n"
                          f"*❌ Escalate if: top1<0.25 OR avg5<0.20 OR hits<2*\n"
                          f"*🤔 Gray zone: LLM decides by intent*")
            
            # Step 2: Call router with real retrieval signals
            router_decision = call_router(user_query, retrieval_signals=retrieval_signals)
            
            action = router_decision.get("action", "escalate")
            confidence = router_decision.get("confidence", 0.0)
            reason = router_decision.get("reason", "")
            
            # Show router decision in sidebar
            with st.sidebar:
                st.info(f"**Router Decision:** {action}\n\n**Confidence:** {confidence:.2f}\n\n**Reason:** {reason}")
            
            # Step 3: Execute action based on router decision
            response_content = ""
            response_metadata = {
                "action": action,
                "confidence": confidence,
                "reason": reason,
            }
            
            if action == "rag":
                # Use RAG tool with all configuration parameters
                rag_tool = create_rag_tool(
                    k_dense=k_dense,
                    k_sparse=k_sparse,
                    top_fuse=top_fuse,
                    top_rerank=top_rerank,
                    neighbor_radius=neighbor_radius,
                    max_context_tokens=max_context_tokens,
                )
                
                with st.spinner("🔍 Searching knowledge base..."):
                    rag_result_str = rag_tool._run(user_query)
                    
                    try:
                        rag_result = json.loads(rag_result_str)
                        
                        # DEBUG: Show RAG result in sidebar
                        with st.sidebar:
                            st.warning(f"**🔬 RAG Debug Info:**\n\n"
                                      f"- can_answer: **{rag_result.get('can_answer', 'N/A')}**\n"
                                      f"- confidence: **{rag_result.get('confidence', 0.0):.2f}**\n"
                                      f"- status: {rag_result.get('status', 'N/A')}\n\n"
                                      f"*Will escalate if: can_answer=False OR confidence<0.5*")
                        
                        if rag_result.get("can_answer", False):
                            response_content = rag_result.get("answer", "No answer provided")
                            response_metadata["sources"] = rag_result.get("sources", [])
                            response_metadata["rag_confidence"] = rag_result.get("confidence", 0.0)
                            
                            # If RAG confidence is too low, escalate
                            if rag_result.get("confidence", 0.0) < 0.5:
                                escalate_tool = create_escalate_tool()
                                escalate_result_str = escalate_tool._run(json.dumps({
                                    "query": user_query,
                                    "reason": "RAG confidence too low"
                                }))
                                escalate_result = json.loads(escalate_result_str)
                                response_content = escalate_result.get("message", response_content)
                                response_metadata["action"] = "escalate"
                                response_metadata["ticket_id"] = escalate_result.get("ticket_id")
                        else:
                            # RAG says it cannot answer, escalate
                            escalate_tool = create_escalate_tool()
                            escalate_result_str = escalate_tool._run(json.dumps({
                                "query": user_query,
                                "reason": rag_result.get("reason", "RAG cannot answer")
                            }))
                            escalate_result = json.loads(escalate_result_str)
                            response_content = escalate_result.get("message", "Unable to answer")
                            response_metadata["action"] = "escalate"
                            response_metadata["ticket_id"] = escalate_result.get("ticket_id")
                    
                    except json.JSONDecodeError:
                        response_content = f"Error parsing RAG response: {rag_result_str[:500]}"
                        response_metadata["error"] = True
            
            elif action == "escalate":
                # Use escalate tool
                escalate_tool = create_escalate_tool()
                with st.spinner("🎫 Creating support ticket..."):
                    escalate_result_str = escalate_tool._run(json.dumps({
                        "query": user_query,
                        "reason": reason
                    }))
                    
                    try:
                        escalate_result = json.loads(escalate_result_str)
                        response_content = escalate_result.get("message", "Support ticket created")
                        response_metadata["ticket_id"] = escalate_result.get("ticket_id")
                    except json.JSONDecodeError:
                        response_content = escalate_result_str
            
            elif action in ["db", "api"]:
                response_content = f"The '{action}' action is not yet implemented. Please contact support for assistance."
                response_metadata["action"] = "escalate"
            
            else:
                response_content = "Unknown action. Please contact support."
                response_metadata["error"] = True
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_content,
            "metadata": response_metadata,
            "timestamp": datetime.now().isoformat()
        })
        
        # Rerun to show the new message
        st.rerun()


if __name__ == "__main__":
    main()

