# 🤖 BoneQuest - Professional RAG WebApp

A complete ChatGPT-like webapp with AI-powered document analysis, chat sessions, admin panel, and voice interaction.

## 🚀 Features

- **Landing Page**: Professional homepage with feature overview
- **Chat Dashboard**: ChatGPT-like interface with session management
- **Admin Panel**: Secure document upload (password: `admin123`)
- **Documentation**: Complete user guide and API reference
- **Voice Interaction**: Speech-to-text and text-to-speech
- **Smart Search**: TF-IDF vector search for document retrieval
- **Session Management**: Save and load chat conversations

## 🏗️ Architecture

- **Frontend**: Single-page HTML app with Tailwind CSS
- **Backend**: FastAPI with RAG functionality
- **AI**: Groq API with GPT-OSS-120B model
- **Vector Search**: Scikit-learn TF-IDF
- **Storage**: Local file system (JSON + Pickle)

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd BoneQuest-RAG
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key" > .env
   ```

3. **Run Application**:
   ```bash
   python backend/main.py
   ```

4. **Access**: Open `http://localhost:8000`

### Render Deployment

1. **Connect Repository** to Render
2. **Set Environment Variable**: `GROQ_API_KEY`
3. **Deploy** using `render.yaml` configuration

### Docker Deployment

```bash
docker build -t bonequest .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key bonequest
```

## 📱 Usage

### For Users
1. Visit the landing page
2. Click "Start Chatting" to access dashboard
3. Create new chat sessions
4. Ask questions about uploaded documents
5. Use voice input for hands-free interaction

### For Admins
1. Navigate to Admin panel
2. Login with password: `admin123`
3. Upload PDF documents
4. Monitor system statistics

## 🔧 API Endpoints

- `GET /` - Landing page
- `POST /chat` - Send chat message
- `POST /chat-speech` - Send speech message (short response)
- `POST /upload` - Upload document (admin only)
- `POST /save-session` - Save chat session
- `GET /sessions` - Get all sessions
- `GET /stats` - System statistics
- `GET /health` - Health check

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Dark Sidebar**: ChatGPT-like navigation
- **Session Management**: Save and load conversations
- **Voice Controls**: Speech-to-text with visual feedback
- **Admin Interface**: Secure document management
- **Documentation**: Built-in help system

## 🔒 Security

- Admin panel protected by password
- CORS enabled for cross-origin requests
- Input validation and error handling
- Secure file upload with type checking

## 📊 Performance

- Efficient TF-IDF vectorization
- Optimized document chunking
- Session persistence
- Health monitoring endpoints

## 🌐 Production Ready

- Environment variable configuration
- Docker containerization
- Render deployment ready
- Error handling and logging
- Health check endpoints

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **AI**: Groq API (GPT-OSS-120B)
- **ML**: Scikit-learn, NumPy
- **Frontend**: HTML5, Tailwind CSS, Vanilla JS
- **Storage**: JSON, Pickle
- **Deployment**: Render, Docker

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

**BoneQuest** - Intelligent Document Analysis Made Simple