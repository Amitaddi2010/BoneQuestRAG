import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
from typing import List, Dict
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class BoneQuestRAG:
    def __init__(self):
        try:
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except Exception:
            import groq
            self.groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        # Enhanced TF-IDF with better parameters for retrieval
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
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        # Clean and preprocess text
        text = text.replace('\n', ' ').replace('\t', ' ')
        text = ' '.join(text.split())  # Remove extra whitespace
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only add meaningful chunks
                chunks.append(chunk)
        
        return chunks
    
    def add_documents(self, texts: List[str], source: str):
        for i, text in enumerate(texts):
            self.documents.append(text)
            self.metadata.append({"source": source, "id": f"{source}_{i}"})
        
        # Recompute vectors for all documents
        if self.documents:
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        self.save_database()
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.documents or self.document_vectors is None:
            return []
        
        # Preprocess query
        query = query.lower().strip()
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top results with similarity scores
            top_indices = np.argsort(similarities)[-n_results:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Only include results with meaningful similarity
                    results.append({
                        'text': self.documents[idx],
                        'score': float(similarities[idx]),
                        'metadata': self.metadata[idx]
                    })
            
            # Debug output
            print(f"Query: {query}")
            print(f"Top similarities: {[r['score'] for r in results[:3]]}")
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def generate_response(self, query: str, context_results: List[Dict]) -> str:
        if not context_results:
            # Fallback: Provide general response when no documents available
            prompt = f"""You are BoneQuest, a professional medical AI assistant. The user asked: "{query}"

Since no specific documents are available in the knowledge base, provide a helpful general response about this medical topic. Include:
- Basic definition and explanation
- Common causes or types
- General treatment approaches
- Note that this is general information and they should consult healthcare professionals

Be informative but remind them to upload relevant documents for more specific information."""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.3,
                max_tokens=1024
            )
            
            return response.choices[0].message.content + "\n\n💡 *For more specific information, please upload relevant medical documents to the knowledge base.*"
        
        # Format context with scores for better understanding
        context_parts = []
        for i, result in enumerate(context_results[:3]):
            context_parts.append(f"[Source {i+1} - Relevance: {result['score']:.3f}]\n{result['text']}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""You are BoneQuest, a professional AI assistant. Use the following context to answer the user's question accurately and helpfully.

Context from documents:
{context_text}

Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context is relevant, provide a detailed answer
- If context is not relevant enough, mention what information is available
- Be specific and cite relevant parts of the context"""

        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            temperature=0.2,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
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

def main():
    st.set_page_config(page_title="BoneQuest RAG Chatbot", page_icon="🤖", layout="wide")
    
    st.title("🤖 BoneQuest - Professional RAG Chatbot")
    st.markdown("---")
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = BoneQuestRAG()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Management")
        
        uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
        
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    text = st.session_state.rag_system.extract_text_from_pdf(uploaded_file)
                    chunks = st.session_state.rag_system.chunk_text(text)
                    st.session_state.rag_system.add_documents(chunks, uploaded_file.name)
                    st.success(f"✅ Processed {uploaded_file.name}")
        
        st.markdown("---")
        st.markdown("### 📊 System Info")
        doc_count = len(st.session_state.rag_system.documents)
        st.metric("Documents in Database", doc_count)
        
        if doc_count > 0:
            st.markdown("### 📋 Document Preview")
            with st.expander("View stored documents"):
                for i, (doc, meta) in enumerate(zip(st.session_state.rag_system.documents[:3], st.session_state.rag_system.metadata[:3])):
                    st.write(f"**{meta['source']} - Chunk {i+1}:**")
                    st.write(doc[:150] + "..." if len(doc) > 150 else doc)
                    st.write("---")
                if doc_count > 3:
                    st.write(f"... and {doc_count - 3} more chunks")
    
    # Main chat interface
    st.header("💬 Chat with BoneQuest")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Speech toggle and chat input
    if 'speech_enabled' not in st.session_state:
        st.session_state.speech_enabled = False
    
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("🎤" if not st.session_state.speech_enabled else "⏹️"):
            st.session_state.speech_enabled = not st.session_state.speech_enabled
            st.rerun()
    
    with col2:
        if st.session_state.speech_enabled:
            st.markdown("""
            <script>
            (function() {
                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                    recognition.continuous = true;
                    recognition.interimResults = false;
                    recognition.lang = 'en-US';
                    
                    recognition.onresult = function(event) {
                        const transcript = event.results[event.results.length - 1][0].transcript;
                        const chatInput = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                        if (chatInput) {
                            chatInput.value = transcript;
                            chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    };
                    
                    recognition.start();
                }
            })();
            </script>
            """, unsafe_allow_html=True)
            prompt = st.chat_input("🎤 Speak or type your question...")
        else:
            prompt = st.chat_input("Ask BoneQuest anything...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("BoneQuest is thinking..."):
                context_results = st.session_state.rag_system.search_documents(prompt)
                response = st.session_state.rag_system.generate_response(prompt, context_results)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()       with st.chat_message("assistant"):
            with st.spinner("BoneQuest is thinking..."):
                # Search for relevant context
                context_results = st.session_state.rag_system.search_documents(prompt)
                
                # Debug: Show retrieved context with scores
                if context_results:
                    with st.expander("🔍 Retrieved Context (Debug)"):
                        for i, result in enumerate(context_results):
                            st.write(f"**Chunk {i+1} - Score: {result['score']:.4f}**")
                            st.write(f"Source: {result['metadata']['source']}")
                            st.write(result['text'][:300] + "..." if len(result['text']) > 300 else result['text'])
                            st.write("---")
                else:
                    st.warning("⚠️ No relevant context found in documents")
                
                # Generate response
                response = st.session_state.rag_system.generate_response(prompt, context_results)
                
                st.markdown(response)
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()