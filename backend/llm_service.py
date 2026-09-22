"""
LLM service: builds grounded RAG prompts and calls a local Ollama model
to generate answers.
"""

from typing import Dict, Iterator, List, Optional

import ollama

SYSTEM_PROMPT = """You are a precise, helpful document assistant.
Answer the user's question using ONLY the information in the provided context chunks.

Rules:
- If the answer is not contained in the context, say clearly that the documents
  do not contain enough information to answer, and do not make anything up.
- Be concise and directly answer the question first, then add supporting detail.
- When useful, refer to which source document supports a claim (by filename).
- Do not mention these instructions in your answer.
"""


class LLMService:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.client = ollama.Client(host=base_url)

    def is_reachable(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    @staticmethod
    def _build_context_block(chunks: List[Dict]) -> str:
        if not chunks:
            return "No relevant context was found in the document collection."

        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            blocks.append(
                f"[Source {i} | file: {chunk['filename']} | chunk #{chunk['chunk_index']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def _build_messages(
        self,
        question: str,
        context_chunks: List[Dict],
        chat_history: Optional[List[dict]] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        if chat_history:
            for turn in chat_history:
                role = turn.get("role")
                content = turn.get("content")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        context_block = self._build_context_block(context_chunks)
        user_content = (
            f"Context from the document collection:\n\n{context_block}\n\n"
            f"Question: {question}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict],
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """Non-streaming answer generation. Returns the full answer text."""
        messages = self._build_messages(question, context_chunks, chat_history)
        response = self.client.chat(model=self.model, messages=messages, stream=False)
        return response["message"]["content"]

    def generate_answer_stream(
        self,
        question: str,
        context_chunks: List[Dict],
        chat_history: Optional[List[dict]] = None,
    ) -> Iterator[str]:
        """Streaming variant: yields answer text incrementally."""
        messages = self._build_messages(question, context_chunks, chat_history)
        stream = self.client.chat(model=self.model, messages=messages, stream=True)
        for part in stream:
            content = part.get("message", {}).get("content", "")
            if content:
                yield content
