import asyncio
import logging
import pyaudio
import sys
import os
import types

# webrtcvad < 2.0.11 requires pkg_resources, which is deprecated in modern python.
# We inject a mock to satisfy its resource_filename call.
if 'pkg_resources' not in sys.modules:
    mock_pkg_resources = types.ModuleType('pkg_resources')
    def resource_filename(package_or_requirement, resource_name):
        import importlib.util
        spec = importlib.util.find_spec(package_or_requirement)
        if spec and spec.submodule_search_locations:
            return os.path.join(spec.submodule_search_locations[0], resource_name)
            
        # Fallback to looking in the virtual environment site-packages
        for path in sys.path:
            if 'site-packages' in path:
                test_path = os.path.join(path, package_or_requirement, resource_name)
                if os.path.exists(test_path):
                    return test_path
        return resource_name
    mock_pkg_resources.resource_filename = resource_filename
    
    class MockDist:
        version = '2.0.10'
    mock_pkg_resources.get_distribution = lambda x: MockDist()
    
    sys.modules['pkg_resources'] = mock_pkg_resources

import webrtcvad

logger = logging.getLogger(__name__)

class AudioIO:
    """Manages audio input and output, including Voice Activity Detection."""
    def __init__(self, sample_rate=16000, chunk_duration_ms=30):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.format = pyaudio.paInt16
        self.channels = 1
        
        self.pa = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(3)  # Aggressiveness 3 (highest)
        
    def _is_speech(self, audio_chunk: bytes) -> bool:
        """Determines if a chunk contains speech using WebRTC VAD."""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

    async def capture_utterance(self, silence_duration_ms=600, stop_event=None) -> bytes:
        """Captures audio until silence is detected, indicating the end of an utterance."""
        stream = self.pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        utterance_frames = []
        silence_frames = 0
        silence_threshold = silence_duration_ms // self.chunk_duration_ms
        in_speech = False
        
        logger.info("Listening for speech...")
        try:
            loop = asyncio.get_running_loop()
            while True:
                if stop_event and stop_event.is_set():
                    break
                # Read audio chunk async to not block
                data = await loop.run_in_executor(None, stream.read, self.chunk_size, False)
                is_speech = self._is_speech(data)
                
                if is_speech:
                    if not in_speech:
                        logger.info("Speech detected.")
                        in_speech = True
                    silence_frames = 0
                    utterance_frames.append(data)
                else:
                    if in_speech:
                        silence_frames += 1
                        utterance_frames.append(data) # keep short silence at end
                        if silence_frames > silence_threshold:
                            logger.info("End of utterance detected.")
                            break
                    else:
                        # Keep a small buffer of pre-speech audio (e.g., 5 frames)
                        if len(utterance_frames) > 5:
                            utterance_frames.pop(0)
                        utterance_frames.append(data)
                        
            return b''.join(utterance_frames)
            
        finally:
            stream.stop_stream()
            stream.close()

    async def play_audio(self, audio_data: bytes, sample_rate: int = 24000, channels: int = 1):
        """Plays back an audio buffer asynchronously."""
        stream = self.pa.open(
            format=self.format,
            channels=channels,
            rate=sample_rate,
            output=True
        )
        try:
            chunk_len = 4096
            loop = asyncio.get_running_loop()
            for i in range(0, len(audio_data), chunk_len):
                chunk = audio_data[i:i+chunk_len]
                await loop.run_in_executor(None, stream.write, chunk)
        finally:
            stream.stop_stream()
            stream.close()

    def cleanup(self):
        """Terminates pyaudio instance."""
        self.pa.terminate()
