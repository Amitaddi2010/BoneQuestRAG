# 🤖 BoneQuest - Professional RAG WebApp

A complete Next.js/FastAPI webapp with claymorphism UI for document analysis and AI chat.

## Architecture

- **Backend**: FastAPI with RAG functionality
- **Frontend**: HTML/JavaScript with Tailwind CSS claymorphism UI
- **AI**: Groq API with GPT-OSS-120B model
- **Vector Search**: TF-IDF with scikit-learn

## Quick Start

1. **Run WebApp**:
   ```bash
   run_webapp.bat
   ```

2. **Manual Setup**:
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   python main.py

   # Frontend
   cd ../frontend
   # Open index.html in browser
   ```

## Features

- **🎨 Claymorphism UI**: Modern soft 3D design
- **🎤 Speech-to-Text**: Voice input with Web Speech API
- **📄 PDF Upload**: Drag & drop document processing
- **💬 Real-time Chat**: Instant AI responses
- **🔍 Vector Search**: Smart document retrieval
- **📊 Live Stats**: Document count and sources

## API Endpoints

- `POST /chat` - Send chat message
- `POST /upload` - Upload PDF document
- `GET /stats` - Get system statistics

## Tech Stack

- **Backend**: FastAPI, Groq, PyPDF2, scikit-learn
- **Frontend**: HTML5, Tailwind CSS, Axios
- **AI**: GPT-OSS-120B via Groq API
- **Speech**: Web Speech API

## Configuration

Set your Groq API key in `.env`:
```
GROQ_API_KEY=your_api_key_here
```