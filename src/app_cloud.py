import streamlit as st
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stt import STT
from src.llm import LLM
from src.tts import TTS
from src.latency_tracker import LatencyTracker

st.set_page_config(page_title="Cloud Voice AI", layout="centered", page_icon="🎙️")

st.title("🎙️ Cloud Voice AI Assistant")
st.markdown("Use this web-native interface for cloud deployments. It safely records audio directly from your browser instead of checking the server microphone.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize pipeline modules purely in session state so they don't block
if "stt" not in st.session_state:
    st.session_state.stt = STT()
    st.session_state.llm = LLM()
    st.session_state.tts = TTS()
    st.session_state.tracker = LatencyTracker()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/wav")

# Native Streamlit audio recorder (works seamlessly via web browsers)
audio_file = st.audio_input("Record a message")

async def process_audio_cloud(audio_bytes):
    tracker = st.session_state.tracker
    tracker.start() # T0
    
    with st.chat_message("user"):
        st.write("Processing audio...")
        tracker.record("T1_STT_START")
        transcript = await st.session_state.stt.transcribe(audio_bytes)
        tracker.record("T2_STT_END")
        st.markdown(f"**You:** {transcript}")
        st.session_state.messages.append({"role": "user", "content": transcript})
    
    if not transcript.strip() or len(transcript) < 2:
        return
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        tracker.record("T3_LLM_START")
        llm_stream = st.session_state.llm.generate_response(transcript)
        
        full_response = ""
        first_token_received = False
        
        async for chunk in llm_stream:
            if not first_token_received:
                tracker.record("T4_LLM_FIRST_TOKEN")
                first_token_received = True
                
            full_response += chunk
            message_placeholder.markdown(full_response + " ▌")
            
        message_placeholder.markdown(full_response)
        
        # After full text response builds out, synthesize TTS
        tracker.record("T6_TTS_START")
        audio_chunk = await st.session_state.tts.synthesize(full_response)
        tracker.record("T7_TTS_FIRST_CHUNK")
        
        if audio_chunk:
            st.audio(audio_chunk, format="audio/wav", autoplay=True)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "audio": audio_chunk
            })
        else:
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        tracker.record("T8_PLAYBACK_START")
        
    tracker.log_breakdown()
    with st.expander("Show Latency Breakdown"):
        st.json(tracker.milestones)

if audio_file:
    st.session_state.last_audio = audio_file.read()
    asyncio.run(process_audio_cloud(st.session_state.last_audio))
    st.rerun()
