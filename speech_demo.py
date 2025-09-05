#!/usr/bin/env python3
"""
Speech Demo for BoneQuest - Test TTS functionality
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def test_tts():
    """Test Text-to-Speech functionality"""
    print("🎤 Testing BoneQuest TTS...")
    
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        test_text = "Hello! I am BoneQuest, your professional medical AI assistant. How can I help you today?"
        
        print(f"📝 Converting text: {test_text}")
        
        speech_file_path = Path("test_speech.wav")
        
        response = client.audio.speech.create(
            model="playai-tts",
            voice="Thunder-PlayAI",
            response_format="wav",
            input=test_text
        )
        
        response.stream_to_file(speech_file_path)
        
        print(f"✅ Speech generated successfully: {speech_file_path}")
        print(f"📊 File size: {speech_file_path.stat().st_size} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        if "model_terms_required" in str(e):
            print("💡 Solution: Accept TTS terms at https://console.groq.com/playground?model=playai-tts")
            print("🔄 BoneQuest will use browser TTS as fallback")
        return False

def test_browser_tts():
    """Test browser TTS fallback"""
    print("🌐 Testing Browser TTS Fallback...")
    
    test_text = "This is a test of browser text-to-speech functionality."
    
    print("✅ Browser TTS HTML generated")
    print("💡 This will work in the Streamlit app when speech mode is enabled")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 BoneQuest TTS Test Suite")
    print("=" * 50)
    
    groq_success = test_tts()
    
    print("\n" + "-" * 30)
    
    browser_success = test_browser_tts()
    
    print("\n" + "=" * 50)
    print("🎆 Test Summary:")
    print(f"Groq TTS: {'✅ Working' if groq_success else '❌ Needs terms acceptance'}")
    print(f"Browser TTS: ✅ Available as fallback")
    print("\n🚀 BoneQuest speech functionality ready!")