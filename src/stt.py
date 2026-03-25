import os
import io
import asyncio
import logging
from groq import AsyncGroq
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class STT:
    def __init__(self, timeout_secs=2.0):
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.timeout_secs = timeout_secs
        
        if self.api_key:
            self.client = AsyncGroq(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GROQ_API_KEY not found. STT will return mock text.")

    async def _groq_transcribe(self, audio_bytes: bytes) -> str:
        if not self.client:
            await asyncio.sleep(0.5) # simulate delay
            return "This is a fallback transcription since no key was provided."
            
        import wave
        audio_io = io.BytesIO()
        with wave.open(audio_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
            
        audio_io.seek(0)
        audio_io.name = "audio.wav"
        
        try:
            translation = await self.client.audio.transcriptions.create(
                file=(audio_io.name, audio_io.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en"
            )
            return translation
        except Exception as e:
            logger.error(f"Groq STT error: {e}")
            raise

    async def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribes with Graceful Degradation / Timeout"""
        try:
            text = await asyncio.wait_for(self._groq_transcribe(audio_bytes), timeout=self.timeout_secs)
            return text
        except asyncio.TimeoutError:
            logger.warning("STT timed out! Falling back to local mock.")
            return "I am sorry, my speech recognition timed out."
        except Exception as e:
            logger.error(f"Failed STT: {e}")
            return "Sorry, I had trouble hearing that."
