import streamlit as st
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def transcribe_audio(audio_path):
    """Send audio to Gemini for transcription"""
    try:
        audio_file = genai.upload_file(audio_path)
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content([
            "Transcribe this doctor-patient conversation verbatim with:\n"
            "1. 'DOCTOR:' and 'PATIENT:' prefixes\n"
            "2. Include pauses and non-verbal cues\n"
            "3. Maintain medical terms accurately",
            audio_file
        ])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

st.title("🏥 Doctor-Patient Conversation Recorder")

# Initialize session state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_chunks' not in st.session_state:
    st.session_state.audio_chunks = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 Start Recording", disabled=st.session_state.recording):
        st.session_state.recording = True
        st.session_state.audio_chunks = []
        st.session_state.start_time = time.time()

with col2:
    if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
        st.session_state.recording = False

# Main recording logic
if st.session_state.recording:
    elapsed = time.time() - st.session_state.start_time
    st.write(f"⏱️ Recording: {int(elapsed)} seconds")
    
    # Continuous recording with pause_threshold
    audio_bytes = audio_recorder(
        text=" ",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        pause_threshold=1800,  # 30 minutes maximum
        key="main_recorder"
    )
    
    if audio_bytes:
        st.session_state.audio_chunks.append(audio_bytes)
        st.experimental_rerun()

# Process after stopping
if not st.session_state.recording and st.session_state.audio_chunks:
    full_audio = b"".join(st.session_state.audio_chunks)
    audio_path = "consultation.wav"
    
    with open(audio_path, "wb") as f:
        f.write(full_audio)
    
    st.audio(full_audio, format="audio/wav")
    
    if st.button("📝 Transcribe Consultation"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_path)
        
        if transcription.startswith("Error"):
            st.error(transcription)
        else:
            st.subheader("Medical Transcript")
            st.markdown(f"```\n{transcription}\n```")
            
            st.download_button(
                label="📥 Download Transcript",
                data=transcription,
                file_name="medical_transcript.txt"
            )
