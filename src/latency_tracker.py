import time
import logging

logger = logging.getLogger(__name__)

class LatencyTracker:
    def __init__(self):
        self.milestones = {}

    def start(self):
        self.milestones.clear()
        self.record("T0_VAD_END")

    def record(self, milestone_name: str):
        self.milestones[milestone_name] = time.time()
        logger.debug(f"Milestone {milestone_name} reached.")

    def log_breakdown(self):
        if "T0_VAD_END" not in self.milestones:
            return
            
        logger.info("\n--- Latency Breakdown ---")
        
        t0 = self.milestones.get("T0_VAD_END")
        
        if "T2_STT_END" in self.milestones:
            stt_time = (self.milestones["T2_STT_END"] - self.milestones.get("T1_STT_START", t0)) * 1000
            logger.info(f"STT Time: {stt_time:.0f}ms")
            
        if "T4_LLM_FIRST_TOKEN" in self.milestones:
            llm_ttft = (self.milestones["T4_LLM_FIRST_TOKEN"] - self.milestones.get("T3_LLM_START", t0)) * 1000
            logger.info(f"LLM TTFT: {llm_ttft:.0f}ms")
            
        if "T7_TTS_FIRST_CHUNK" in self.milestones:
            tts_ttfb = (self.milestones["T7_TTS_FIRST_CHUNK"] - self.milestones.get("T6_TTS_START", t0)) * 1000
            logger.info(f"TTS TFB: {tts_ttfb:.0f}ms")
            
        if "T8_PLAYBACK_START" in self.milestones:
            ttfa = (self.milestones["T8_PLAYBACK_START"] - t0) * 1000
            logger.info(f"Total TTFA (End-of-Speech to First-Audio): {ttfa:.0f}ms")
            
        logger.info("-------------------------\n")
