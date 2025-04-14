import streamlit as st
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def transcribe_audio(audio_path):
    """Send audio to Gemini for transcription"""
    try:
        # Upload audio file to Gemini
        audio_file = genai.upload_file(audio_path)
        
        # Use Gemini Pro model
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        
        response = model.generate_content(
            ["Transcribe this audio verbatim into text. Include punctuation and capitalization.", audio_file]
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

st.title("Voice Recorder with Gemini Transcription")

# Audio recorder
audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_size="2x",
    pause_threshold=10.0,
)

if audio_bytes:
    # Save recording
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    
    # Display audio
    st.audio(audio_bytes, format="audio/wav")
    
    # Transcribe button
    if st.button("Transcribe Audio"):
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_path)
        
        if transcription.startswith("Error"):
            st.error(transcription)
        else:
            st.subheader("Transcription:")
            st.write(transcription)
            
            # Optional: Save transcription
            with open("transcription.txt", "w") as f:
                f.write(transcription)
            st.download_button(
                label="Download Transcription",
                data=transcription,
                file_name="transcription.txt"
            )
