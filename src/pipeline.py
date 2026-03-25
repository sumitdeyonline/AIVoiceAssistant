import asyncio
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_io import AudioIO
from src.stt import STT
from src.llm import LLM
from src.tts import TTS
from src.latency_tracker import LatencyTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self, ui_queue=None, stop_event=None):
        self.audio_io = AudioIO()
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        self.tracker = LatencyTracker()
        self.ui_queue = ui_queue
        self.stop_event = stop_event

    def _ui_put(self, msg_type, content):
        if self.ui_queue:
            self.ui_queue.put({"type": msg_type, "content": content})

    async def run(self):
        logger.info("Voice Assistant Started. Speak into the microphone. Press Ctrl+C to stop.")
        self._ui_put("status", "System Ready. Speak to the assistant!")
        try:
            while True:
                if self.stop_event and self.stop_event.is_set():
                    logger.info("Stop event detected. Exiting pipeline.")
                    break
                    
                self._ui_put("status", "Listening...")
                audio_bytes = await self.audio_io.capture_utterance(stop_event=self.stop_event)
                
                if self.stop_event and self.stop_event.is_set():
                    break
                if not audio_bytes or len(audio_bytes) < 1000:
                    continue
                    
                self._ui_put("status", "Hearing you...")
                self.tracker.start() # T0: VAD END
                
                self.tracker.record("T1_STT_START")
                transcript = await self.stt.transcribe(audio_bytes)
                self.tracker.record("T2_STT_END")
                
                logger.info(f"User heard: {transcript}")
                self._ui_put("user", transcript)
                
                if not transcript.strip() or len(transcript) < 2:
                    self._ui_put("status", "Listening...")
                    continue

                self._ui_put("status", "Thinking...")
                self.tracker.record("T3_LLM_START")
                llm_stream = self.llm.generate_response(transcript)
                
                sentence_buffer = ""
                full_assistant_response = ""
                first_token_received = False
                
                print("Assistant: ", end="", flush=True)
                
                async for chunk in llm_stream:
                    if not first_token_received:
                        self.tracker.record("T4_LLM_FIRST_TOKEN")
                        first_token_received = True
                        
                    sentence_buffer += chunk
                    full_assistant_response += chunk
                    print(chunk, end="", flush=True)
                    self._ui_put("assistant_chunk", full_assistant_response)
                    
                    # Synthesize completed sentences immediately
                    if any(p in chunk for p in ['.', '!', '?']):
                        if "T5_LLM_FIRST_SENTENCE" not in self.tracker.milestones:
                            self.tracker.record("T5_LLM_FIRST_SENTENCE")
                            
                        sentence = sentence_buffer.strip()
                        sentence_buffer = ""
                        
                        if sentence:
                            if "T6_TTS_START" not in self.tracker.milestones:
                                self.tracker.record("T6_TTS_START")
                            self._ui_put("status", "Speaking...")
                            audio_chunk = await self.tts.synthesize(sentence)
                            if "T7_TTS_FIRST_CHUNK" not in self.tracker.milestones:
                                self.tracker.record("T7_TTS_FIRST_CHUNK")
                                
                            if audio_chunk:
                                if "T8_PLAYBACK_START" not in self.tracker.milestones:
                                    self.tracker.record("T8_PLAYBACK_START")
                                await self.audio_io.play_audio(audio_chunk)

                if sentence_buffer.strip():
                    self._ui_put("status", "Speaking...")
                    audio_chunk = await self.tts.synthesize(sentence_buffer.strip())
                    if audio_chunk:
                        await self.audio_io.play_audio(audio_chunk)
                        
                print("\n")
                self._ui_put("assistant_done", full_assistant_response)
                self.tracker.log_breakdown()
                self._ui_put("latency", self.tracker.milestones)
                
        except KeyboardInterrupt:
            logger.info("Stopping assistant...")
            self._ui_put("status", "Assistant Stopped.")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._ui_put("status", f"Error: {e}")
        finally:
            self.audio_io.cleanup()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    asyncio.run(assistant.run())
