"""Escalate Tool: Create support ticket when RAG cannot answer."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain.tools import BaseTool
from pydantic import Field


class EscalateTool(BaseTool):
    """Tool for escalating queries to human support when RAG cannot provide answers.
    
    This tool creates a support ticket and notifies the help desk.
    """

    name: str = "escalate_to_support"
    description: str = (
        "Use this tool when you cannot answer the user's question with available information, "
        "or when the confidence is too low, or when the question requires human expertise. "
        "Input should be a JSON string with 'query' and 'reason' fields."
    )
    
    # Configuration
    ticket_log_path: Path = Field(
        default=Path("logs/tickets.jsonl"),
        description="Path to store ticket logs"
    )

    def _run(self, query_info: str) -> str:
        """Create a support ticket and log it."""
        try:
            # Parse input (could be plain text or JSON)
            if query_info.strip().startswith('{'):
                try:
                    info = json.loads(query_info)
                    query = info.get("query", query_info)
                    reason = info.get("reason", "Unable to provide confident answer")
                except json.JSONDecodeError:
                    query = query_info
                    reason = "Unable to provide confident answer"
            else:
                query = query_info
                reason = "Unable to provide confident answer"
            
            # Generate ticket
            ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
            ticket = {
                "ticket_id": ticket_id,
                "query": query,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "status": "open",
                "assigned_to": "help_desk",
            }
            
            # Log ticket
            self.ticket_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ticket_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
            
            # Return formatted response
            response = {
                "status": "escalated",
                "ticket_id": ticket_id,
                "message": (
                    f"I apologize, but I don't have sufficient information to answer your question confidently. "
                    f"I've created a support ticket ({ticket_id}) and notified our help desk. "
                    f"A support specialist will contact you shortly to assist with your inquiry."
                ),
                "query": query,
                "reason": reason,
            }
            
            return json.dumps(response, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Failed to create support ticket: {str(e)}",
                "fallback_message": (
                    "I apologize, but I'm unable to answer your question and encountered an error "
                    "creating a support ticket. Please contact support directly at support@company.com "
                    "or call our help desk."
                ),
            }, ensure_ascii=False)

    async def _arun(self, query_info: str) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(query_info)


def create_escalate_tool(ticket_log_path: Optional[Path] = None) -> EscalateTool:
    """Factory function to create a configured escalate tool."""
    kwargs = {}
    if ticket_log_path:
        kwargs["ticket_log_path"] = ticket_log_path
    return EscalateTool(**kwargs)

