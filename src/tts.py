import os
import io
import asyncio
import logging
import subprocess
from dotenv import load_dotenv
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

class TTS:
    def __init__(self, timeout_secs=2.0):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.timeout_secs = timeout_secs
        
        if self.api_key and AsyncOpenAI:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OPENAI_API_KEY not found. TTS will fall back to Mac 'say' command.")

    async def _openai_synthesize(self, text: str) -> bytes:
        if not self.client:
            raise ValueError("No TTS client")
        
        try:
            response = await self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="wav" # pyaudio supports wav (pcm) processing nicely
            )
            return response.content
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            raise

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio. Returns wav bytes. If fails/timeout, falls back to local macOS 'say' command."""
        if not self.client:
            await self._local_fallback(text)
            return b"" 

        try:
            audio_bytes = await asyncio.wait_for(self._openai_synthesize(text), timeout=self.timeout_secs)
            return audio_bytes
        except asyncio.TimeoutError:
            logger.warning("TTS timed out! Falling back to local TTS.")
            await self._local_fallback(text)
            return b""
        except Exception as e:
            logger.error(f"TTS failed: {e}. Falling back to local TTS.")
            await self._local_fallback(text)
            return b""
            
    async def _local_fallback(self, text: str):
        """Uses macOS built-in 'say'."""
        logger.info(f"Local TTS speaking: {text}")
        process = await asyncio.create_subprocess_exec(
            'say', text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.communicate()
