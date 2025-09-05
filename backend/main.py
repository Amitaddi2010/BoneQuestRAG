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
import PyPDF2
from typing import List, Dict
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import base64

load_dotenv()

app = FastAPI(title="BoneQuest RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    context: List[Dict]

class SessionRequest(BaseModel):
    session_id: str
    messages: List[Dict]

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    source = Column(String(255))
    doc_id = Column(String(255))
    timestamp = Column(DateTime)

class VectorizerData(Base):
    __tablename__ = 'vectorizer_data'
    id = Column(Integer, primary_key=True)
    vectorizer_pickle = Column(LargeBinary)
    vectors_pickle = Column(LargeBinary)

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True)
    messages = Column(Text)
    timestamp = Column(DateTime)

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
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            strip_accents='ascii'
        )
        self.documents = []
        self.document_vectors = None
        self.metadata = []
        
        # Database setup
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            database_url = "sqlite:///bonequest.db"  # Fallback for local
        
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.db_session = Session()
        
        self.load_database()
        
    def extract_text_from_pdf(self, pdf_bytes) -> str:
        text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
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
        # Limit total documents to prevent memory issues
        max_docs = 2000
        
        for i, text in enumerate(texts):
            self.documents.append(text)
            self.metadata.append({"source": source, "id": f"{source}_{i}", "timestamp": datetime.now().isoformat()})
        
        # Clean up if too many documents
        if len(self.documents) > max_docs:
            excess = len(self.documents) - max_docs
            self.documents = self.documents[excess:]
            self.metadata = self.metadata[excess:]
        
        try:
            if self.documents:
                self.document_vectors = self.vectorizer.fit_transform(self.documents)
            self.save_database()
        except Exception as e:
            print(f"Database save error: {e}")
            # Reset if corrupted
            self.documents = []
            self.metadata = []
            self.document_vectors = None
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.documents or self.document_vectors is None:
            print(f"No documents available. Total docs: {len(self.documents)}")
            return []
        
        query = query.lower().strip()
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.005:  # Lower threshold
                    results.append({
                        'text': self.documents[idx],
                        'score': float(similarities[idx]),
                        'metadata': self.metadata[idx]
                    })
            
            print(f"Search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def generate_response(self, query: str, context_results: List[Dict]) -> str:
        if not self.groq_client:
            if not context_results:
                return f"I'm BoneQuest AI assistant. You asked: '{query}'. Please upload documents for more specific information, or I can provide general guidance on this topic."
            
            context_parts = []
            for i, result in enumerate(context_results[:3]):
                context_parts.append(f"[Source {i+1}]\n{result['text']}")
            
            context_text = "\n\n".join(context_parts)
            context_display = "\n\n".join([f"**Source {i+1}:**\n{result['text'][:800]}..." for i, result in enumerate(context_results[:3])])
            return f"Based on your documents, here's what I found about '{query}':\n\n{context_display}\n\nThis information comes from your uploaded documents."
            
        return self._generate_complete_response(query, context_results)
    
    def _generate_complete_response(self, query: str, context_results: List[Dict]) -> str:
        """Generate complete response with cutoff detection and retry"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                max_tokens = 2500 + (attempt * 1000)
                
                if not context_results:
                    prompt = f"""You are BoneQuest AI, a medical assistant specializing in bone and skeletal conditions.

User Question: {query}

Provide a complete, well-structured response. Use headings, bullet points, and clear formatting. Ensure your response ends naturally with a complete sentence or conclusion."""
                else:
                    context_parts = []
                    for i, result in enumerate(context_results[:3]):
                        context_parts.append(f"[Source {i+1}]\n{result['text']}")
                    
                    context_text = "\n\n".join(context_parts)
                    
                    prompt = f"""You are BoneQuest AI, a medical assistant. Based on the provided medical documents, give a complete response.

Medical Context:
{context_text}

User Question: {query}

Provide a comprehensive response based on the context. Use clear structure with headings and bullet points. Ensure your response ends naturally with a complete sentence or conclusion."""
                
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="openai/gpt-oss-120b",
                    temperature=0.2,
                    max_tokens=max_tokens,
                    stop=None
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
            # Delete existing session
            self.db_session.query(ChatSession).filter_by(session_id=session_id).delete()
            
            # Save new session
            session = ChatSession(
                session_id=session_id,
                messages=json.dumps(messages),
                timestamp=datetime.now()
            )
            self.db_session.add(session)
            self.db_session.commit()
        except Exception as e:
            print(f"Save session error: {e}")
            self.db_session.rollback()
    
    def load_sessions(self) -> Dict:
        try:
            sessions = self.db_session.query(ChatSession).all()
            result = {}
            for session in sessions:
                result[session.session_id] = {
                    'messages': json.loads(session.messages),
                    'timestamp': session.timestamp.isoformat()
                }
            return result
        except Exception as e:
            print(f"Load sessions error: {e}")
            return {}
    
    def save_database(self):
        try:
            # Clear existing data
            self.db_session.query(Document).delete()
            self.db_session.query(VectorizerData).delete()
            
            # Save documents
            for i, doc in enumerate(self.documents):
                doc_record = Document(
                    text=doc,
                    source=self.metadata[i].get('source', ''),
                    doc_id=self.metadata[i].get('id', ''),
                    timestamp=datetime.fromisoformat(self.metadata[i].get('timestamp', datetime.now().isoformat()))
                )
                self.db_session.add(doc_record)
            
            # Save vectorizer and vectors
            if self.document_vectors is not None:
                vectorizer_data = VectorizerData(
                    vectorizer_pickle=pickle.dumps(self.vectorizer),
                    vectors_pickle=pickle.dumps(self.document_vectors)
                )
                self.db_session.add(vectorizer_data)
            
            self.db_session.commit()
            print(f"Database saved: {len(self.documents)} documents")
        except Exception as e:
            print(f"Save database error: {e}")
            self.db_session.rollback()
    
    def load_database(self):
        try:
            # Load documents
            docs = self.db_session.query(Document).all()
            self.documents = [doc.text for doc in docs]
            self.metadata = [{
                'source': doc.source,
                'id': doc.doc_id,
                'timestamp': doc.timestamp.isoformat()
            } for doc in docs]
            
            # Load vectorizer and vectors
            vectorizer_data = self.db_session.query(VectorizerData).first()
            if vectorizer_data:
                self.vectorizer = pickle.loads(vectorizer_data.vectorizer_pickle)
                self.document_vectors = pickle.loads(vectorizer_data.vectors_pickle)
            
            print(f"Database loaded: {len(self.documents)} documents")
        except Exception as e:
            print(f"Load database error: {e}")
            self.documents = []
            self.document_vectors = None
            self.metadata = []

rag_system = BoneQuestRAG()

@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    context_results = rag_system.search_documents(request.query)
    response = rag_system.generate_response(request.query, context_results)
    
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

@app.get("/stats")
async def get_stats():
    return {
        "document_count": len(rag_system.documents),
        "sources": len(set([meta.get('source', '') for meta in rag_system.metadata])),
        "vectorizer_ready": rag_system.document_vectors is not None,
        "sample_docs": [doc[:100] + "..." for doc in rag_system.documents[:3]]
    }

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