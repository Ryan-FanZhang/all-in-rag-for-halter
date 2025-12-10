from __future__ import annotations

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage


def generate_answer(
    messages: List[BaseMessage],
    model_name: str = "gpt-5",
    temperature: float = 1,
):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    return llm.invoke(messages)

