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
import datetime
import json
from pathlib import Path
import base64
import time

load_dotenv()

# Page config
st.set_page_config(
    page_title="BoneQuest - AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



class BoneQuestRAG:
    def __init__(self):
        try:
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        except Exception:
            import groq
            self.groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
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
            self.metadata.append({"source": source, "id": f"{source}_{i}", "timestamp": datetime.datetime.now().isoformat()})
        
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
    
    def generate_response_speech(self, query: str, context_results: List[Dict]) -> str:
        """Generate precise, point-to-point responses for speech mode"""
        if not context_results:
            prompt = f"""You are BoneQuest. The user asked: "{query}"

Provide a precise, concise answer in 2-3 sentences maximum. Be direct and to the point for speech delivery."""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content
        
        context_parts = []
        for i, result in enumerate(context_results[:2]):
            context_parts.append(f"Source {i+1}: {result['text'][:200]}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""Context: {context_text}

Question: {query}

Provide a precise, direct answer in 2-3 sentences maximum. Focus on key facts only."""

        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=200
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

# Session management
def save_chat_session(session_name, messages):
    sessions_file = "chat_sessions.json"
    sessions = {}
    if os.path.exists(sessions_file):
        try:
            with open(sessions_file, 'r') as f:
                sessions = json.load(f)
        except:
            sessions = {}
    
    sessions[session_name] = {
        'messages': messages,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(sessions_file, 'w') as f:
        json.dump(sessions, f)

def load_chat_sessions():
    sessions_file = "chat_sessions.json"
    if os.path.exists(sessions_file):
        try:
            with open(sessions_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = BoneQuestRAG()

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_session' not in st.session_state:
    st.session_state.current_session = 'New Chat'

if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

if 'speech_mode' not in st.session_state:
    st.session_state.speech_mode = False

# Sidebar navigation
with st.sidebar:
    st.header("🤖 BoneQuest")
    st.caption("AI RAG Assistant")
    
    # Speech Mode Toggle
    st.subheader("🎤 Speech Mode")
    
    speech_toggle = st.toggle("Enable Speech", value=st.session_state.speech_mode)
    if speech_toggle != st.session_state.speech_mode:
        st.session_state.speech_mode = speech_toggle
        st.rerun()
    
    if st.session_state.speech_mode:
        st.success("🎤 **Speech Mode Active**")
        
        # Initialize speech state
        if 'is_listening' not in st.session_state:
            st.session_state.is_listening = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Voice Chat", disabled=st.session_state.is_listening):
                st.session_state.is_listening = True
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Voice Chat", disabled=not st.session_state.is_listening):
                st.session_state.is_listening = False
                st.rerun()
        
        if st.session_state.is_listening:
            st.info("🎤 Voice chat active - Speak your question")
            
            # Voice interface with Streamlit components only
            st.markdown("<div style='background: #f0f8ff; padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;'>", unsafe_allow_html=True)
            
            st.write("**Status:** Ready to listen")
            
            if st.button("🎤 Click & Speak", key="voice_btn"):
                # Simple JavaScript without inline handlers
                st.markdown("""
                <script>
                (function() {
                    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                        recognition.continuous = false;
                        recognition.interimResults = false;
                        recognition.lang = 'en-US';
                        
                        recognition.onresult = function(event) {
                            const transcript = event.results[0][0].transcript;
                            const chatInput = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
                            if (chatInput) {
                                chatInput.value = transcript;
                                chatInput.dispatchEvent(new Event('input', { bubbles: true }));
                                setTimeout(() => {
                                    const submitBtn = window.parent.document.querySelector('button[data-testid="stChatInputSubmitButton"]');
                                    if (submitBtn) submitBtn.click();
                                }, 300);
                            }
                        };
                        
                        recognition.start();
                    }
                })();
                </script>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Navigation
    page = st.selectbox("Navigate", 
                       ["🏠 Home", "💬 Chat", "📚 Documentation", "🔧 Admin Panel", "📊 Sessions"])
    
    st.session_state.current_page = page.split(" ", 1)[1]

# Main content based on selected page
if st.session_state.current_page == "Home":
    # Landing Page
    st.title("🤖 BoneQuest")
    st.subheader("Professional AI-Powered RAG Assistant")
    st.write("Intelligent Document Analysis & Question Answering System")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## Welcome to BoneQuest")
        st.markdown("""
        BoneQuest is a state-of-the-art Retrieval-Augmented Generation (RAG) system that helps you 
        extract insights from your documents using advanced AI technology.
        """)
        
        if st.button("🚀 Start Chatting", type="primary", use_container_width=True):
            st.session_state.current_page = "Chat"
            st.rerun()
    
    # Features
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📄 Document Processing**
        
        Upload and process PDF documents with intelligent text extraction and chunking.
        """)
    
    with col2:
        st.markdown("""
        **🔍 Smart Search**
        
        Advanced vector-based semantic search for finding relevant information quickly.
        """)
    
    with col3:
        st.markdown("""
        **🤖 AI Responses**
        
        Powered by GPT-OSS-120B for accurate, context-aware answers.
        """)
    
    # Stats
    doc_count = len(st.session_state.rag_system.documents)
    sources = len(set([meta.get('source', '') for meta in st.session_state.rag_system.metadata]))
    
    st.markdown("## 📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Document Chunks", doc_count)
    
    with col2:
        st.metric("Source Files", sources)
    
    with col3:
        st.metric("Vector Search", "TF-IDF")
    
    with col4:
        st.metric("AI Model", "GPT-OSS-120B")

elif st.session_state.current_page == "Chat":
    # Chat Interface
    st.title("💬 Chat with BoneQuest")
    
    # Speech Mode Indicator
    if st.session_state.speech_mode:
        st.success("🎤 **Speech Mode Active** - Responses will be spoken aloud")
    
    # Session management in sidebar
    with st.sidebar:
        st.markdown("### 💾 Chat Sessions")
        
        sessions = load_chat_sessions()
        session_names = ["New Chat"] + list(sessions.keys())
        
        selected_session = st.selectbox("Select Session", session_names)
        
        if selected_session != st.session_state.current_session:
            if selected_session == "New Chat":
                st.session_state.messages = []
            else:
                st.session_state.messages = sessions[selected_session]['messages']
            st.session_state.current_session = selected_session
        
        if st.button("💾 Save Current Session"):
            if st.session_state.messages:
                session_name = f"Chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                save_chat_session(session_name, st.session_state.messages)
                st.success(f"Session saved as {session_name}")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.current_session = "New Chat"
            st.rerun()
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask BoneQuest anything... (or use voice input above)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # Show wave animation if speech mode is active
            if st.session_state.speech_mode:
                st.info("🎤 Generating speech response...")
            
            with st.spinner("BoneQuest is thinking..."):
                context_results = st.session_state.rag_system.search_documents(prompt)
                
                # Show debug info in chat interface
                if context_results:
                    with st.expander("🔍 Retrieved Context (Debug)"):
                        for i, result in enumerate(context_results):
                            st.write(f"**Chunk {i+1} - Score: {result['score']:.4f}**")
                            st.write(f"Source: {result['metadata']['source']}")
                            st.write(result['text'][:200] + "..." if len(result['text']) > 200 else result['text'])
                            st.write("---")
                
                # Use speech-optimized response if speech mode is active
                if st.session_state.speech_mode:
                    response = st.session_state.rag_system.generate_response_speech(prompt, context_results)
                else:
                    response = st.session_state.rag_system.generate_response(prompt, context_results)
                st.markdown(response)
                
                # Generate speech if speech mode is enabled
                if st.session_state.speech_mode:
                    # Clean text for speech
                    clean_text = response[:300].replace('"', "'").replace('\n', ' ').replace('\r', ' ')
                    # Remove markdown
                    import re
                    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
                    clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
                    clean_text = re.sub(r'[#*`]', '', clean_text)
                    
                    st.markdown(f"""
                    <script>
                    (function() {{
                        if ('speechSynthesis' in window) {{
                            const utterance = new SpeechSynthesisUtterance("{clean_text}");
                            utterance.rate = 0.9;
                            utterance.pitch = 1.0;
                            utterance.volume = 0.8;
                            speechSynthesis.speak(utterance);
                        }}
                    }})();
                    </script>
                    """, unsafe_allow_html=True)
                    st.info("🔊 Speaking response...")
                
                st.session_state.messages.append({"role": "assistant", "content": response})

elif st.session_state.current_page == "Documentation":
    # Documentation Page
    st.title("📚 Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Getting Started", "API Reference", "FAQ"])
    
    with tab1:
        st.markdown("""
        ## 🚀 Getting Started with BoneQuest
        
        ### Step 1: Upload Documents
        1. Go to the Admin Panel
        2. Upload PDF documents using the file uploader
        3. Click "Process Document" to add them to the knowledge base
        
        ### Step 2: Start Chatting
        1. Navigate to the Chat page
        2. Ask questions about your uploaded documents
        3. BoneQuest will search for relevant context and provide answers
        
        ### Step 3: Manage Sessions
        1. Save important conversations using the "Save Session" button
        2. Load previous sessions from the dropdown menu
        3. Clear chat history when needed
        """)
    
    with tab2:
        st.markdown("""
        ## 🔧 API Reference
        
        ### Core Components
        
        **BoneQuestRAG Class**
        - `extract_text_from_pdf()`: Extract text from PDF files
        - `chunk_text()`: Split text into manageable chunks
        - `add_documents()`: Add documents to the vector database
        - `search_documents()`: Perform semantic search
        - `generate_response()`: Generate AI responses with context
        
        **Vector Search**
        - Uses TF-IDF vectorization for document similarity
        - Cosine similarity for ranking relevant chunks
        - Returns top 3 most relevant document chunks
        
        **AI Model**
        - Model: OpenAI GPT-OSS-120B
        - Provider: Groq API
        - Temperature: 0.3 for consistent responses
        """)
    
    with tab3:
        st.markdown("""
        ## ❓ Frequently Asked Questions
        
        **Q: What file formats are supported?**
        A: Currently, BoneQuest supports PDF documents only.
        
        **Q: How does the RAG system work?**
        A: BoneQuest extracts text from your documents, creates vector embeddings, and uses semantic search to find relevant context for your questions.
        
        **Q: Can I delete uploaded documents?**
        A: Yes, use the Admin Panel to manage your document database.
        
        **Q: Are my conversations saved?**
        A: Chat sessions are saved locally and can be managed through the Sessions page.
        
        **Q: What AI model does BoneQuest use?**
        A: BoneQuest uses OpenAI GPT-OSS-120B via the Groq API for fast, accurate responses.
        """)

elif st.session_state.current_page == "Admin Panel":
    # Admin Panel
    st.title("🔧 Admin Panel")
    
    # Simple authentication
    if not st.session_state.admin_authenticated:
        st.markdown("### 🔐 Admin Authentication")
        password = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password == "admin123":  # Simple password - change in production
                st.session_state.admin_authenticated = True
                st.success("Authentication successful!")
                st.rerun()
            else:
                st.error("Invalid password")
    else:
        st.success("✅ Authenticated as Admin")
        
        tab1, tab2 = st.tabs(["Document Management", "System Settings"])
        
        with tab1:
            st.markdown("### 📄 Document Upload")
            
            uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
            
            if uploaded_file:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        text = st.session_state.rag_system.extract_text_from_pdf(uploaded_file)
                        chunks = st.session_state.rag_system.chunk_text(text)
                        st.session_state.rag_system.add_documents(chunks, uploaded_file.name)
                        st.success(f"✅ Processed {uploaded_file.name}")
            
            st.markdown("### 📊 Knowledge Base Status")
            doc_count = len(st.session_state.rag_system.documents)
            st.metric("Total Document Chunks", doc_count)
            
            if doc_count > 0:
                st.markdown("### 📋 Document Preview")
                for i, (doc, meta) in enumerate(zip(st.session_state.rag_system.documents[:5], st.session_state.rag_system.metadata[:5])):
                    with st.expander(f"{meta['source']} - Chunk {i+1}"):
                        st.write(doc[:300] + "..." if len(doc) > 300 else doc)
                
                if st.button("🗑️ Clear All Documents", type="secondary"):
                    if st.checkbox("I confirm I want to delete all documents"):
                        st.session_state.rag_system.documents = []
                        st.session_state.rag_system.document_vectors = None
                        st.session_state.rag_system.metadata = []
                        st.session_state.rag_system.save_database()
                        st.success("All documents cleared!")
                        st.rerun()
        
        with tab2:
            st.markdown("### ⚙️ System Configuration")
            st.info("System settings and configuration options will be available in future updates.")
            
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

elif st.session_state.current_page == "Sessions":
    # Sessions Management
    st.title("📊 Chat Sessions")
    
    sessions = load_chat_sessions()
    
    if not sessions:
        st.info("No saved chat sessions found.")
    else:
        st.markdown(f"### 💾 Saved Sessions ({len(sessions)})")
        
        for session_name, session_data in sessions.items():
            with st.expander(f"📝 {session_name} - {session_data['timestamp'][:19]}"):
                st.markdown(f"**Messages:** {len(session_data['messages'])}")
                
                if st.button(f"Load {session_name}", key=f"load_{session_name}"):
                    st.session_state.messages = session_data['messages']
                    st.session_state.current_session = session_name
                    st.session_state.current_page = "Chat"
                    st.rerun()
                
                # Show first few messages
                for i, msg in enumerate(session_data['messages'][:4]):
                    role_icon = "👤" if msg['role'] == 'user' else "🤖"
                    st.write(f"{role_icon} **{msg['role'].title()}:** {msg['content'][:100]}...")
                
                if len(session_data['messages']) > 4:
                    st.write(f"... and {len(session_data['messages']) - 4} more messages")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>BoneQuest v2.0 - Professional RAG Assistant | Powered by Groq & GPT-OSS-120B</div>", unsafe_allow_html=True)