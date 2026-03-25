import streamlit as st
import queue
import threading
import time
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import VoiceAssistant

st.set_page_config(page_title="Streaming Voice AI", layout="centered", page_icon="🎙️")

st.title("🎙️ Streaming Voice AI Assistant")
st.markdown("Ultra-low latency conversational AI with fallback degradation.")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ui_queue" not in st.session_state:
    st.session_state.ui_queue = queue.Queue()
if "assistant_thread" not in st.session_state:
    st.session_state.assistant_thread = None
if "status" not in st.session_state:
    st.session_state.status = "Idle"
if "current_chunk" not in st.session_state:
    st.session_state.current_chunk = ""
if "last_latency" not in st.session_state:
    st.session_state.last_latency = {}
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()

def run_assistant(q, stop_event):
    """Background thread to run the async pipeline."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    assistant = VoiceAssistant(ui_queue=q, stop_event=stop_event)
    try:
        loop.run_until_complete(assistant.run())
    except Exception as e:
        q.put({"type": "status", "content": f"Thread Error: {e}"})
    finally:
        loop.close()

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    is_running = st.session_state.assistant_thread and st.session_state.assistant_thread.is_alive()
    
    if not is_running:
        if st.button("🟢 Start Listening", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_latency = {}
            st.session_state.status = "Starting..."
            st.session_state.stop_event.clear()
            thread = threading.Thread(target=run_assistant, args=(st.session_state.ui_queue, st.session_state.stop_event), daemon=True)
            st.session_state.assistant_thread = thread
            thread.start()
            st.rerun()
    else:
        if st.button("🔴 Stop Conversation", use_container_width=True):
            st.session_state.stop_event.set()
            st.session_state.status = "Stopping..."
            st.rerun()
            
    st.markdown("---")
    st.markdown(f"**Status:** `{st.session_state.status}`")
    
    if st.session_state.last_latency:
        st.markdown("### Latency Milestones")
        t0 = st.session_state.last_latency.get("T0_VAD_END")
        if t0:
            stt_end = st.session_state.last_latency.get("T2_STT_END", t0)
            llm_first = st.session_state.last_latency.get("T4_LLM_FIRST_TOKEN", t0)
            tts_first = st.session_state.last_latency.get("T7_TTS_FIRST_CHUNK", t0)
            
            st.metric("STT Time", f"{int((stt_end - t0)*1000)} ms")
            st.metric("LLM Time-to-First-Token", f"{int((llm_first - stt_end)*1000)} ms")
            st.metric("TTS Time-to-First-Byte", f"{int((tts_first - llm_first)*1000)} ms")
            st.metric("Total Time-to-First-Audio", f"{int((tts_first - t0)*1000)} ms")

# Drain the queue
while not st.session_state.ui_queue.empty():
    event = st.session_state.ui_queue.get()
    evt_type = event.get("type")
    content = event.get("content")
    
    if evt_type == "status":
        st.session_state.status = content
    elif evt_type == "user":
        st.session_state.messages.append({"role": "user", "content": content})
    elif evt_type == "assistant_chunk":
        st.session_state.current_chunk = content
    elif evt_type == "assistant_done":
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.session_state.current_chunk = ""
    elif evt_type == "latency":
        st.session_state.last_latency = content

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Render current streaming chunk if any
if st.session_state.current_chunk:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.current_chunk + " ▌")

# Continuous Polling Loop
if st.session_state.assistant_thread and st.session_state.assistant_thread.is_alive():
    time.sleep(0.1)
    st.rerun()
