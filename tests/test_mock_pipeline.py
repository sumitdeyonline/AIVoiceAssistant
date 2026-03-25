import sys
import os
import asyncio
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import VoiceAssistant

logging.basicConfig(level=logging.INFO)

async def run_test():
    assistant = VoiceAssistant()
    
    # Mock capture_utterance to return 1 sec of zeroes, and then stop the loop
    call_count = 0
    async def mock_capture():
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return b'\x00' * (16000 * 2)
        else:
            await asyncio.sleep(0.1)
            raise KeyboardInterrupt()

    assistant.audio_io.capture_utterance = mock_capture
    
    # Mock play_audio so we don't output to a missing audio device
    async def mock_play(audio_data):
        pass
    assistant.audio_io.play_audio = mock_play

    # Set very short timeouts to ensure we hit graceful degradation paths
    assistant.stt.timeout_secs = 0.1
    assistant.llm.timeout_secs = 0.1
    assistant.tts.timeout_secs = 0.1

    print("--- Running Mocked Pipeline Test ---")
    try:
        await assistant.run()
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
        
    print("--- Test finished successfully! ---")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(run_test()))
