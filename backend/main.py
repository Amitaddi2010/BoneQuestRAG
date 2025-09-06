from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
try:
    from groq import Groq
except ImportError:
    print("Groq import failed, using fallback")
    Groq = None
from typing import List, Dict
import PyPDF2
import io
import json
from datetime import datetime
# Removed PostgreSQL imports for Vercel compatibility

load_dotenv()

app = FastAPI(title="BoneQuest RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Dict] = []

class ChatResponse(BaseModel):
    response: str
    context: List[Dict]

class SessionRequest(BaseModel):
    session_id: str
    messages: List[Dict]

# Simplified storage for Vercel

class BoneQuestRAG:
    def __init__(self):
        if Groq is None:
            print("Groq not available, using fallback")
            self.groq_client = None
        else:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("No GROQ_API_KEY found in environment")
                self.groq_client = None
            else:
                try:
                    self.groq_client = Groq(api_key=api_key)
                    print(f"Groq client initialized successfully")
                except Exception as e:
                    print(f"Groq client error: {e}")
                    self.groq_client = None
        
        self.documents = []
        self.metadata = []
        
        # Simple file storage
        self.db_file = "bonequest_db.json"
        self.sessions_file = "chat_sessions.json"
        self.load_database()
        
    def extract_text_from_pdf(self, pdf_bytes) -> str:
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = ' '.join(text.split())
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
        
        return chunks
    
    def add_documents(self, texts: List[str], source: str):
        for i, text in enumerate(texts):
            self.documents.append(text)
            self.metadata.append({"source": source, "id": f"{source}_{i}", "timestamp": datetime.now().isoformat()})
        
        self.save_database()
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        query = query.lower().strip()
        results = []
        
        # Simple keyword matching
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            score = 0
            
            # Count keyword matches
            for word in query.split():
                if word in doc_lower:
                    score += doc_lower.count(word)
            
            if score > 0:
                results.append({
                    'text': doc,
                    'score': score / len(query.split()),
                    'metadata': self.metadata[i] if i < len(self.metadata) else {}
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    def generate_response(self, query: str, context_results: List[Dict], chat_history: List[Dict] = None) -> str:
        if not self.groq_client:
            if not context_results:
                return f"I'm BoneQuest AI assistant. You asked: '{query}'. Please upload documents for more specific information, or I can provide general guidance on this topic."
            
            context_parts = []
            for i, result in enumerate(context_results[:3]):
                context_parts.append(f"[Source {i+1}]\n{result['text']}")
            
            context_text = "\n\n".join(context_parts)
            context_display = "\n\n".join([f"📄 **Source {i+1}** (Score: {result['score']:.2f}):\n{result['text'][:300]}..." for i, result in enumerate(context_results[:2])])
            return f"🔍 **RAG Response** - Based on your uploaded documents:\n\n{context_display}\n\n✅ This information comes from your knowledge base."
            
        return self._generate_complete_response(query, context_results, chat_history)
    
    def _generate_complete_response(self, query: str, context_results: List[Dict], chat_history: List[Dict] = None) -> str:
        """Generate complete response with cutoff detection and retry"""
        max_retries = 2
        
        # Extract length requirements from query
        length_indicators = {
            'in 1 line': 1,
            'in 2 lines': 2, 
            'in 3 lines': 3,
            'in 2-3 lines': 3,
            'briefly': 2,
            'short': 2,
            'summarize': 3,
            'quick': 1
        }
        
        max_lines = None
        query_lower = query.lower()
        for indicator, lines in length_indicators.items():
            if indicator in query_lower:
                max_lines = lines
                break
        
        for attempt in range(max_retries):
            try:
                # Adjust tokens based on length requirement
                if max_lines and max_lines <= 3:
                    max_tokens = 80 + (attempt * 20)  # Very short responses
                else:
                    max_tokens = 800 + (attempt * 200)  # Normal responses
                
                # Build chat history context
                history_context = ""
                if chat_history and len(chat_history) > 0:
                    recent_history = chat_history[-4:]  # Last 4 messages for context
                    history_parts = []
                    for msg in recent_history:
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        history_parts.append(f"{role}: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"{role}: {msg['content']}")
                    history_context = f"\n\nPrevious conversation context:\n" + "\n".join(history_parts) + "\n"
                
                if not context_results:
                    if max_lines:
                        prompt = f"""STRICT INSTRUCTION: Answer in EXACTLY {max_lines} line(s). Do not exceed this limit.{history_context}
Current Question: {query}

Response format: "💡 **General Response** - [Your {max_lines} line answer here]"

IMPORTANT: Stop after {max_lines} line(s). No additional text."""
                    else:
                        prompt = f"""You are BoneQuest AI, a medical assistant specializing in bone and skeletal conditions.{history_context}
User Question: {query}

Start with "💡 **General Response** - " and provide a complete, well-structured response. Use headings, bullet points, and clear formatting. Ensure your response ends naturally with a complete sentence or conclusion."""
                else:
                    context_parts = []
                    for i, result in enumerate(context_results[:2]):
                        truncated_text = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                        context_parts.append(f"📄 Source {i+1} (Score: {result['score']:.2f}):\n{truncated_text}")
                    
                    context_text = "\n\n".join(context_parts)
                    
                    if max_lines:
                        prompt = f"""STRICT INSTRUCTION: Answer in EXACTLY {max_lines} line(s). Do not exceed this limit.{history_context}
Medical Context:
{context_text}

Current Question: {query}

Response format: "🔍 **RAG Response** - [Your {max_lines} line answer here] ✅ Knowledge base."

IMPORTANT: Stop after {max_lines} line(s). No additional text."""
                    else:
                        prompt = f"""You are BoneQuest AI, a medical assistant. Based on the provided medical documents, give a complete response.{history_context}
Medical Context:
{context_text}

User Question: {query}

Start with "🔍 **RAG Response** - Based on your uploaded documents:" and provide a comprehensive response based on the context. Use clear structure with headings and bullet points. End with "✅ This information comes from your knowledge base." Ensure your response ends naturally with a complete sentence or conclusion."""
                
                # Add stop tokens for short responses (max 4 items)
                stop_tokens = None
                if max_lines and max_lines <= 3:
                    stop_tokens = ["\n\n", "Additionally", "Furthermore", "Moreover"]
                
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=max_tokens,
                    stop=stop_tokens
                )
                
                response_text = response.choices[0].message.content.strip()
                
                if self._is_response_complete(response_text):
                    return response_text
                
                if attempt < max_retries - 1:
                    continue
                else:
                    return response_text
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error generating response: {str(e)}"
                continue
        
        return "Unable to generate complete response after multiple attempts."
    
    def _is_response_complete(self, response: str) -> bool:
        """Check if response appears complete"""
        if not response:
            return False
        
        response = response.strip()
        
        if response.endswith(('.', '!', '?', ')', ']', '}')):
            last_sentence = response.split('.')[-2] if '.' in response else response
            
            incomplete_words = ['and', 'or', 'but', 'however', 'therefore', 'additionally', 'furthermore', 'moreover', 'consequently', 'thus', 'hence', 'also', 'including', 'such', 'like', 'with', 'without', 'through', 'during', 'after', 'before']
            
            words = last_sentence.lower().split()
            if words and words[-1] in incomplete_words:
                return False
            
            return True
        
        if len(response) < 50:
            return False
        
        if response[-1].isalnum() and ' ' not in response[-10:]:
            return False
        
        return False
    
    def generate_speech_response(self, query: str, context_results: List[Dict]) -> str:
        if not self.groq_client:
            return "AI service unavailable."
            
        if not context_results:
            prompt = f"""Answer in 1-2 sentences only. User asked: "{query}"

Be concise and direct for speech delivery."""
            
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="openai/gpt-oss-120b",
                    temperature=0.1,
                    max_tokens=100
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error: {str(e)}"
        
        context_parts = []
        for i, result in enumerate(context_results[:2]):
            context_parts.append(f"Source {i+1}: {result['text'][:200]}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""Context: {context_text}

Question: {query}

Answer in 1-2 sentences only. Be precise and direct for speech."""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def save_session(self, session_id: str, messages: List[Dict]):
        try:
            sessions = self.load_sessions()
            sessions[session_id] = {
                'messages': messages,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f)
        except Exception as e:
            print(f"Save session error: {e}")
    
    def load_sessions(self) -> Dict:
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Load sessions error: {e}")
                return {}
        return {}
    
    def delete_session(self, session_id: str):
        try:
            sessions = self.load_sessions()
            if session_id in sessions:
                del sessions[session_id]
                with open(self.sessions_file, 'w') as f:
                    json.dump(sessions, f)
                return True
            return False
        except Exception as e:
            print(f"Delete session error: {e}")
            return False
    
    def save_database(self):
        try:
            data = {
                'documents': self.documents,
                'metadata': self.metadata
            }
            with open(self.db_file, 'w') as f:
                json.dump(data, f)
            print(f"Database saved: {len(self.documents)} documents")
        except Exception as e:
            print(f"Save database error: {e}")
    
    def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                print(f"Database loaded: {len(self.documents)} documents")
            except Exception as e:
                print(f"Load database error: {e}")
                self.documents = []
                self.metadata = []
        else:
            print("No database file found")

rag_system = BoneQuestRAG()

@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    context_results = rag_system.search_documents(request.query)
    response = rag_system.generate_response(request.query, context_results, request.chat_history)
    
    return ChatResponse(
        response=response,
        context=context_results
    )

@app.post("/chat-speech", response_model=ChatResponse)
async def chat_speech(request: QueryRequest):
    context_results = rag_system.search_documents(request.query)
    response = rag_system.generate_speech_response(request.query, context_results)
    
    return ChatResponse(
        response=response,
        context=context_results
    )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    pdf_bytes = await file.read()
    text = rag_system.extract_text_from_pdf(pdf_bytes)
    chunks = rag_system.chunk_text(text)
    rag_system.add_documents(chunks, file.filename)
    
    return {"message": f"Document {file.filename} processed successfully"}

@app.post("/save-session")
async def save_session(request: SessionRequest):
    rag_system.save_session(request.session_id, request.messages)
    return {"message": "Session saved successfully"}

@app.get("/sessions")
async def get_sessions():
    return rag_system.load_sessions()

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    success = rag_system.delete_session(session_id)
    if success:
        return {"message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/stats")
async def get_stats():
    try:
        sources = set()
        if rag_system.metadata:
            sources = set([meta.get('source', '') for meta in rag_system.metadata if meta.get('source')])
        
        return {
            "document_count": len(rag_system.documents),
            "sources": len(sources),
            "sample_docs": [doc[:100] + "..." for doc in rag_system.documents[:3]] if rag_system.documents else []
        }
    except Exception as e:
        return {"document_count": 0, "sources": 0, "sample_docs": [], "error": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "groq_available": rag_system.groq_client is not None,
        "documents_loaded": len(rag_system.documents),
        "vectorizer_fitted": rag_system.document_vectors is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)