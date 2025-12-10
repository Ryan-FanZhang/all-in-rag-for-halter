"""Memory Manager: Manages conversation history using LangChain."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class MemoryManager:
    """Manages conversation memory for the AI agent.
    
    Features:
    - Stores conversation history
    - Provides context to tools
    - Supports multi-turn conversations
    - Handles follow-up questions
    """
    
    def __init__(self, k: int = 5):
        """Initialize memory manager.
        
        Args:
            k: Number of recent conversation turns to remember (a turn = user + assistant)
        """
        self.window_size = k
        # Use deque for efficient windowed storage (stores LangChain Message objects)
        self.windowed_messages: deque = deque(maxlen=k * 2)  # k turns = k*2 messages
        # Full history with metadata
        self.full_history: List[Dict[str, Any]] = []
    
    def add_user_message(self, message: str, metadata: Optional[Dict] = None) -> None:
        """Add a user message to memory."""
        user_msg = HumanMessage(content=message)
        self.windowed_messages.append(user_msg)
        
        # Store in full history with metadata
        self.full_history.append({
            "role": "user",
            "content": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def add_ai_message(self, message: str, metadata: Optional[Dict] = None) -> None:
        """Add an AI message to memory."""
        ai_msg = AIMessage(content=message)
        self.windowed_messages.append(ai_msg)
        
        # Store in full history with metadata
        self.full_history.append({
            "role": "assistant",
            "content": message,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get recent N conversation turns."""
        return self.full_history[-n*2:] if self.full_history else []
    
    def get_context_string(self, include_system: bool = True) -> str:
        """Get conversation history as a formatted string.
        
        Args:
            include_system: Whether to include system context
            
        Returns:
            Formatted conversation history
        """
        context_parts = []
        if include_system:
            context_parts.append("## Conversation History:")
        
        for msg in self.windowed_messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_context_for_llm(self, current_query: str, system_prompt: str = "") -> List[Any]:
        """Get conversation context formatted for LLM input.
        
        Args:
            current_query: Current user query
            system_prompt: Optional system prompt to prepend
            
        Returns:
            List of messages for LLM
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        # Add conversation history from windowed messages
        messages.extend(list(self.windowed_messages))
        
        # Add current query
        messages.append(HumanMessage(content=current_query))
        
        return messages
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.windowed_messages.clear()
        self.full_history = []
    
    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.full_history:
            return "No conversation history."
        
        total_turns = len(self.full_history) // 2
        recent = self.get_recent_history(3)
        
        summary = f"Total conversation turns: {total_turns}\n\n"
        summary += "Recent exchanges:\n"
        for msg in recent[-6:]:  # Last 3 turns (6 messages)
            role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary += f"{role}: {content}\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dictionary for persistence."""
        return {
            "full_history": self.full_history,
            "window_size": self.window_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], k: int = 5) -> "MemoryManager":
        """Restore memory from dictionary.
        
        Args:
            data: Serialized memory data
            k: Window size for buffer
            
        Returns:
            Restored MemoryManager instance
        """
        window_size = data.get("window_size", k)
        manager = cls(k=window_size)
        manager.full_history = data.get("full_history", [])
        
        # Restore windowed messages from full history
        # Only restore the last k*2 messages to maintain window
        recent_history = manager.full_history[-(window_size * 2):]
        for msg in recent_history:
            if msg["role"] == "user":
                manager.windowed_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                manager.windowed_messages.append(AIMessage(content=msg["content"]))
        
        return manager


def create_memory_manager(window_size: int = 5) -> MemoryManager:
    """Factory function to create a memory manager.
    
    Args:
        window_size: Number of recent conversation turns to remember
        
    Returns:
        MemoryManager instance
    """
    return MemoryManager(k=window_size)

