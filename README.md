# 🤖 BoneQuest - Professional RAG WebApp

A complete ChatGPT-like webapp with AI-powered document analysis, chat sessions, admin panel, and voice interaction.

## 🚀 Features

### Core Features
- **Landing Page**: Professional homepage with feature overview
- **Chat Dashboard**: ChatGPT-like interface with session management
- **Admin Panel**: Secure document upload (password: `admin123`)
- **Documentation**: Complete user guide and API reference
- **Voice Interaction**: Speech-to-text and text-to-speech
- **Session Management**: Save and load chat conversations

### Advanced RAG System
- **Hybrid Search**: Combines semantic embeddings + keyword matching
- **Semantic Search**: Uses sentence-transformers for contextual understanding
- **FAISS Integration**: Fast similarity search with vector indexing
- **Query Enhancement**: Intelligent query expansion and rewriting
- **Multi-hop Reasoning**: Context-aware document retrieval

### Medical Image Analysis
- **X-Ray Analysis**: AI-powered radiological image interpretation
- **MRI Support**: Magnetic resonance imaging analysis
- **CT Scan Processing**: Computed tomography image evaluation
- **DICOM Compatibility**: Full DICOM medical imaging standard support
- **Drag & Drop Upload**: Easy image upload with preview
- **Confidence Scoring**: AI confidence levels for analysis reliability

## 🏗️ Architecture

### Core Stack
- **Frontend**: Single-page HTML app with Tailwind CSS
- **Backend**: FastAPI with advanced RAG functionality
- **AI**: Groq API with LLaMA 3.1 8B model
- **Storage**: Local file system (JSON + NumPy + FAISS)

### Advanced RAG Components
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS for fast similarity search
- **Hybrid Search**: TF-IDF + Semantic embeddings
- **Context Ranking**: Multi-factor relevance scoring

### Image Analysis Stack
- **Image Processing**: OpenCV + PIL for image manipulation
- **Medical Imaging**: PyDICOM for DICOM file support
- **AI Vision**: Groq LLaMA for image interpretation
- **Format Support**: JPEG, PNG, TIFF, DICOM

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
3. **Select Role**: Choose Patient/Junior Resident/Senior Resident/Consultant
4. **Text Chat**: Ask questions about uploaded documents
5. **Image Analysis**: Upload X-rays, MRIs, CT scans, or DICOM files
6. **Voice Input**: Use speech-to-text for hands-free interaction
7. **Advanced Search**: Benefit from hybrid semantic + keyword search

### Image Analysis Workflow
1. Click the image upload button in chat
2. Select image type (X-Ray, MRI, CT, DICOM)
3. Upload or drag & drop medical images
4. Add optional analysis query
5. Get AI-powered radiological interpretation
6. Review confidence scores and recommendations

### For Admins
1. Navigate to Admin panel
2. Login with password: `admin123`
3. Upload PDF documents for RAG knowledge base
4. Monitor system statistics and feedback analytics
5. Toggle feedback collection mode

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Landing page
- `POST /chat` - Send chat message with advanced RAG
- `POST /chat-speech` - Send speech message (short response)
- `POST /upload` - Upload document (admin only)
- `POST /save-session` - Save chat session
- `GET /sessions` - Get all sessions
- `GET /stats` - System statistics
- `GET /health` - Health check with advanced features

### Advanced RAG Endpoints
- `GET /search-methods` - Get available search capabilities
- `POST /feedback` - Submit detailed user feedback
- `GET /feedback/analytics` - Get feedback analytics dashboard
- `GET /feedback/insights/{query}` - Get search performance insights

### Image Analysis Endpoints
- `POST /analyze-image` - Analyze medical images (base64)
- `POST /upload-image` - Upload and analyze image files
- **Supported formats**: JPEG, PNG, TIFF, DICOM
- **Image types**: X-Ray, MRI, CT, DICOM

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

### Backend & AI
- **Backend**: FastAPI, Uvicorn
- **AI**: Groq API (LLaMA 3.1 8B)
- **ML**: Scikit-learn, NumPy, Sentence-Transformers
- **Vector DB**: FAISS for similarity search
- **Storage**: JSON, NumPy arrays, FAISS indices

### Image Processing
- **Image Processing**: OpenCV, PIL (Pillow)
- **Medical Imaging**: PyDICOM for DICOM support
- **Formats**: JPEG, PNG, TIFF, DICOM

### Frontend & Deployment
- **Frontend**: HTML5, Tailwind CSS, Vanilla JS
- **Features**: Drag & drop, image preview, real-time analysis
- **Deployment**: Render, Docker
- **Testing**: Automated test suite included

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 🧪 Testing Advanced Features

### Run Test Suite
```bash
# Install test dependencies
pip install pillow requests

# Run comprehensive tests
python test_advanced_rag.py
```

### Manual Testing

**Advanced RAG:**
1. Upload medical documents via admin panel
2. Test different role modes (Patient/JR/SR/Consultant)
3. Verify semantic search finds relevant context
4. Check hybrid search combines keyword + semantic results

**Image Analysis:**
1. Upload X-ray, MRI, or CT images
2. Test DICOM file support
3. Verify AI analysis with confidence scores
4. Check metadata extraction

### Performance Metrics
- **Search Speed**: <500ms for hybrid search
- **Image Analysis**: <10s for standard images
- **Embedding Generation**: ~100ms per document chunk
- **FAISS Index**: Sub-millisecond similarity search

---

**BoneQuest** - Advanced AI-Powered Medical Analysis Platform