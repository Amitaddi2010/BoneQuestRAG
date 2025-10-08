#!/usr/bin/env python3
"""
Test script for Advanced BoneQuest RAG system
Tests semantic search, hybrid search, and image analysis capabilities
"""

import requests
import base64
import json
import os
from PIL import Image
import io

# Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test if the server is running with advanced features"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Server Health Check:")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Groq Available: {data.get('groq_available')}")
            print(f"   - Documents Loaded: {data.get('documents_loaded')}")
            print(f"   - Semantic Search: {data.get('semantic_search')}")
            print(f"   - Image Analysis: {data.get('image_analysis')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_search_methods():
    """Test available search methods"""
    try:
        response = requests.get(f"{BASE_URL}/search-methods")
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Search Methods Available:")
            print(f"   - Semantic Search: {data.get('semantic_search')}")
            print(f"   - Hybrid Search: {data.get('hybrid_search')}")
            print(f"   - Keyword Search: {data.get('keyword_search')}")
            print(f"   - Image Analysis: {data.get('image_analysis')}")
            return True
        else:
            print(f"❌ Search methods check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search methods error: {e}")
        return False

def test_chat_functionality():
    """Test advanced chat with different modes"""
    test_queries = [
        {"query": "What are the contraindications for hip replacement?", "mode": "normal"},
        {"query": "Explain the surgical approach for ACL reconstruction", "mode": "jr"},
        {"query": "Discuss complications in spinal fusion surgery", "mode": "sr"},
        {"query": "Latest evidence on knee arthroscopy outcomes", "mode": "consultant"}
    ]
    
    print("\n✅ Testing Chat Functionality:")
    
    for i, test in enumerate(test_queries):
        try:
            response = requests.post(f"{BASE_URL}/chat", json={
                "query": test["query"],
                "mode": test["mode"],
                "chat_history": []
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"   {i+1}. Mode '{test['mode']}': ✅ Response received ({len(data['response'])} chars)")
                print(f"      Context sources: {len(data.get('context', []))}")
            else:
                print(f"   {i+1}. Mode '{test['mode']}': ❌ Failed ({response.status_code})")
                
        except Exception as e:
            print(f"   {i+1}. Mode '{test['mode']}': ❌ Error: {e}")

def create_test_image():
    """Create a simple test medical image"""
    # Create a simple grayscale image that looks like an X-ray
    img = Image.new('L', (512, 512), color=20)  # Dark background
    
    # Add some simple shapes to simulate bone structures
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Simulate femur bone
    draw.ellipse([200, 100, 300, 400], fill=200, outline=255)
    # Simulate joint
    draw.ellipse([180, 380, 320, 450], fill=180, outline=255)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_image_analysis():
    """Test medical image analysis functionality"""
    print("\n✅ Testing Image Analysis:")
    
    try:
        # Create test image
        test_image_bytes = create_test_image()
        image_base64 = base64.b64encode(test_image_bytes).decode('utf-8')
        
        # Test image analysis
        response = requests.post(f"{BASE_URL}/analyze-image", json={
            "image_data": image_base64,
            "image_type": "xray",
            "query": "Analyze this X-ray image",
            "mode": "normal"
        })
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Image analysis successful:")
            print(f"      - Analysis length: {len(data['analysis'])} chars")
            print(f"      - Findings count: {len(data['findings'])}")
            print(f"      - Confidence: {data['confidence']:.2f}")
            print(f"      - Metadata keys: {list(data['image_metadata'].keys())}")
            return True
        else:
            print(f"   ❌ Image analysis failed: {response.status_code}")
            if response.text:
                print(f"      Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Image analysis error: {e}")
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("\n✅ Testing File Upload:")
    
    try:
        # Create test image file
        test_image_bytes = create_test_image()
        
        files = {
            'file': ('test_xray.png', test_image_bytes, 'image/png')
        }
        
        response = requests.post(f"{BASE_URL}/upload-image", 
                               files=files, 
                               data={'image_type': 'xray'})
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ File upload successful:")
            print(f"      - Filename: {data['filename']}")
            print(f"      - Analysis length: {len(data['analysis'])} chars")
            print(f"      - Confidence: {data['confidence']:.2f}")
            return True
        else:
            print(f"   ❌ File upload failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ File upload error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 BoneQuest Advanced RAG System Tests")
    print("=" * 50)
    
    # Test basic connectivity
    if not test_health_check():
        print("\n❌ Server not accessible. Please start the server first.")
        return
    
    # Test search capabilities
    test_search_methods()
    
    # Test chat functionality
    test_chat_functionality()
    
    # Test image analysis
    test_image_analysis()
    
    # Test file upload
    test_file_upload()
    
    print("\n" + "=" * 50)
    print("🎉 Advanced RAG System Tests Complete!")
    print("\nTo start the server, run: python backend/main.py")
    print("Then access the web interface at: http://localhost:8000")

if __name__ == "__main__":
    main()