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
        # Upload audio file to Gemini
        audio_file = genai.upload_file(audio_path)
        
        # Use Gemini Flash model (faster for audio)
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        response = model.generate_content(
            [
                "You are a medical transcription specialist. Transcribe this doctor-patient conversation verbatim with these requirements:",
                "1. Format as dialogue with 'DOCTOR:' and 'PATIENT:' prefixes",
                "2. Include all pauses, filler words, and non-verbal cues like [coughs]",
                "3. Maintain medical terminology accuracy",
                "4. Add relevant timestamps every 30 seconds",
                audio_file
            ]
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

st.title("🏥 Doctor-Patient Conversation Recorder")

# Session state for recording management
if 'recording_start' not in st.session_state:
    st.session_state.recording_start = None
if 'audio_chunks' not in st.session_state:
    st.session_state.audio_chunks = []

# Recording controls
col1, col2 = st.columns(2)
with col1:
    if st.button("🎤 Start Recording", disabled=st.session_state.recording_start is not None):
        st.session_state.recording_start = time.time()
        st.session_state.audio_chunks = []
        st.rerun()

with col2:
    if st.button("⏹️ Stop Recording", disabled=st.session_state.recording_start is None):
        st.session_state.recording_start = None
        st.rerun()

# Recording status and timer
if st.session_state.recording_start:
    elapsed = time.time() - st.session_state.recording_start
    st.write(f"⏱️ Recording: {int(elapsed)} seconds")
    
    # Record in 30-second chunks to avoid memory issues
    if int(elapsed) % 30 == 0 and int(elapsed) > 0:
        st.warning("Saving 30-second chunk...")
        audio_bytes = audio_recorder(
            pause_threshold=30.0,
            key=f"recorder_{int(elapsed)}"
        )
        if audio_bytes:
            st.session_state.audio_chunks.append(audio_bytes)

# Process complete recording
if st.session_state.audio_chunks and st.session_state.recording_start is None:
    full_audio = b"".join(st.session_state.audio_chunks)
    audio_path = "consultation.wav"
    
    with open(audio_path, "wb") as f:
        f.write(full_audio)
    
    st.audio(full_audio, format="audio/wav")
    
    if st.button("📝 Transcribe Consultation"):
        with st.spinner("Transcribing... (This may take a few minutes)"):
            transcription = transcribe_audio(audio_path)
        
        if transcription.startswith("Error"):
            st.error(transcription)
        else:
            st.subheader("Medical Transcript")
            st.markdown(f"```\n{transcription}\n```", unsafe_allow_html=True)
            
            # Save options
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Transcript",
                    data=transcription,
                    file_name="medical_transcript.txt",
                    mime="text/plain"
                )
            with col2:
                if st.button("🗑️ Clear Session"):
                    st.session_state.audio_chunks = []
                    st.rerun()

# Instructions
st.sidebar.markdown("""
## Instructions
1. Click **Start Recording** to begin
2. Speak clearly (doctor and patient)
3. Click **Stop Recording** when finished
4. Press **Transcribe** to generate transcript

## Tips for Best Results:
- Record in a quiet environment
- Place microphone midway between speakers
- Speak at normal volume
- Pause briefly between speakers
""")
