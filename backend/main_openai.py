from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import PyPDF2
from typing import List, Dict
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import openai

load_dotenv()

app = FastAPI(title="BoneQuest RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    context: List[Dict]

class BoneQuestRAG:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY") or "demo"
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
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
        self.db_file = "bonequest_db.pkl"
        self.load_database()
        
    def extract_text_from_pdf(self, pdf_bytes) -> str:
        text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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
            self.metadata.append({"source": source, "id": f"{source}_{i}"})
        
        if self.documents:
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        self.save_database()
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.documents or self.document_vectors is None:
            return []
        
        query = query.lower().strip()
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:
                    results.append({
                        'text': self.documents[idx],
                        'score': float(similarities[idx]),
                        'metadata': self.metadata[idx]
                    })
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def generate_response(self, query: str, context_results: List[Dict]) -> str:
        if not context_results:
            return f"I'm BoneQuest AI assistant. You asked: '{query}'. Please upload documents for more specific information, or I can provide general guidance on this topic."
        
        context_parts = []
        for i, result in enumerate(context_results[:3]):
            context_parts.append(f"[Source {i+1}]\n{result['text']}")
        
        context_text = "\n\n".join(context_parts)
        
        response = f"Based on your documents, here's what I found about '{query}':\n\n{context_text[:500]}...\n\nThis information comes from your uploaded documents. For more detailed analysis, please ask specific questions about the content."
        
        return response
    
    def save_database(self):
        data = {
            'documents': self.documents,
            'document_vectors': self.document_vectors,
            'metadata': self.metadata,
            'vectorizer': self.vectorizer
        }
        with open(self.db_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load_database(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.document_vectors = data.get('document_vectors', None)
                self.metadata = data.get('metadata', [])
                if 'vectorizer' in data:
                    self.vectorizer = data['vectorizer']
            except:
                self.documents = []
                self.document_vectors = None
                self.metadata = []

rag_system = BoneQuestRAG()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    context_results = rag_system.search_documents(request.query)
    response = rag_system.generate_response(request.query, context_results)
    
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

@app.get("/stats")
async def get_stats():
    return {
        "document_count": len(rag_system.documents),
        "sources": len(set([meta.get('source', '') for meta in rag_system.metadata]))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)