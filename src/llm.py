import os
import asyncio
import logging
from typing import AsyncGenerator
from groq import AsyncGroq
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, timeout_secs=3.0):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.timeout_secs = timeout_secs
        self.system_prompt = "You are a helpful, extremely concise voice assistant. Always reply in one or two short sentences."
        
        if self.api_key:
            self.client = AsyncGroq(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GROQ_API_KEY not found. LLM will return mock text.")

    async def _groq_stream(self, text: str) -> AsyncGenerator[str, None]:
        if not self.client:
            await asyncio.sleep(0.5)
            yield "This is a mock LLM response. "
            await asyncio.sleep(0.2)
            yield "I don't have an API key."
            return

        try:
            stream = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                model="llama-3.1-8b-instant",
                stream=True
            )
            
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Groq LLM error: {e}")
            yield "Sorry, my brain stopped working."

    async def generate_response(self, text: str) -> AsyncGenerator[str, None]:
        """Streams LLM tokens, with an initial wait_for timeout on the first yield."""
        stream = self._groq_stream(text)
        
        try:
            first_chunk = await asyncio.wait_for(stream.__anext__(), timeout=self.timeout_secs)
            yield first_chunk
        except asyncio.TimeoutError:
            logger.warning("LLM timed out waiting for TTFT! Yielding fallback.")
            yield "Let me think... "
            return
        except StopAsyncIteration:
            return
            
        async for chunk in stream:
            yield chunk
