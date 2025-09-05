#!/usr/bin/env python3
"""
Audio Demo for BoneQuest - Test Speech-to-Text functionality
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def test_whisper_stt():
    """Test Speech-to-Text functionality with Whisper"""
    print("🎤 Testing BoneQuest Speech-to-Text...")
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Check if test audio file exists
        test_audio = Path("test_audio.wav")
        if not test_audio.exists():
            print("❌ No test audio file found.")
            print("💡 Create a test_audio.wav file or use the web interface to upload audio.")
            return False
        
        print(f"📁 Using audio file: {test_audio}")
        print(f"📊 File size: {test_audio.stat().st_size} bytes")
        
        # Transcribe using Whisper
        with open(test_audio, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(test_audio.name, file.read()),
                model="whisper-large-v3-turbo",
                response_format="text"
            )
        
        print(f"✅ Transcription successful!")
        print(f"📝 Result: {transcription}")
        
        return True
        
    except Exception as e:
        print(f"❌ STT Error: {e}")
        if "model_terms_required" in str(e):
            print("💡 Solution: Accept Whisper terms at Groq console")
        return False

def create_sample_instructions():
    """Provide instructions for creating test audio"""
    print("\n" + "="*50)
    print("🎯 How to Test Speech-to-Text:")
    print("="*50)
    print("1. Record a short audio message (5-10 seconds)")
    print("2. Save it as 'test_audio.wav' in this directory")
    print("3. Run this script again")
    print("\n📱 Mobile Users:")
    print("- Use voice recorder app")
    print("- Export as WAV/MP3")
    print("- Transfer to computer")
    print("\n🖥️ Desktop Users:")
    print("- Use Windows Voice Recorder")
    print("- Use Audacity (free)")
    print("- Use online voice recorders")
    print("\n🌐 Web Interface:")
    print("- Use BoneQuest's file upload feature")
    print("- Upload audio files directly in the app")

if __name__ == "__main__":
    print("🤖 BoneQuest Audio Testing Suite")
    print("="*50)
    
    success = test_whisper_stt()
    
    if not success:
        create_sample_instructions()
    
    print("\n🚀 BoneQuest speech functionality ready for web interface!")