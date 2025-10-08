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
from typing import List, Dict, Optional
import PyPDF2
import io
import json
from datetime import datetime
import base64
import numpy as np
# Advanced RAG imports
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("Advanced RAG libraries not available")
    SentenceTransformer = None
    faiss = None
# Image analysis imports
try:
    from PIL import Image
    import cv2
    import pydicom
    import matplotlib.pyplot as plt
except ImportError:
    print("Image analysis libraries not available")
    Image = None
    cv2 = None
    pydicom = None
    plt = None

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
    mode: str = "normal"  # "normal" or "advanced"
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # AI model to use

class ChatResponse(BaseModel):
    response: str
    context: List[Dict]

class SessionRequest(BaseModel):
    session_id: str
    messages: List[Dict]

class FeedbackRequest(BaseModel):
    message_id: str
    query: str
    response: str
    feedback_type: str  # 'positive', 'negative', 'correction'
    rating: int  # 1-5 scale
    comment: str = ""
    correction: str = ""
    context_quality: int = 1  # 1-5 scale
    response_accuracy: int = 1  # 1-5 scale
    response_completeness: int = 1  # 1-5 scale
    search_relevance: int = 1  # 1-5 scale
    timestamp: str = ""

class FeedbackAnalytics(BaseModel):
    total_feedback: int
    avg_rating: float
    positive_count: int
    negative_count: int
    correction_count: int
    common_issues: List[Dict]
    improvement_suggestions: List[str]

class ImageAnalysisRequest(BaseModel):
    image_data: str  # base64 encoded image
    image_type: str  # "xray", "mri", "ct", "dicom"
    query: str = "Analyze this medical image"
    mode: str = "normal"
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # AI model to use

class ImageAnalysisResponse(BaseModel):
    analysis: str
    findings: List[str]
    recommendations: str
    confidence: float
    image_metadata: Dict

# Simplified storage for Vercel

class AdvancedBoneQuestRAG:
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
        self.embeddings = None
        self.faiss_index = None
        
        # Initialize semantic search
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Semantic search initialized")
            except Exception as e:
                print(f"Semantic search init error: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # File storage
        self.db_file = "bonequest_db.json"
        self.sessions_file = "chat_sessions.json"
        self.feedback_file = "feedback_data.json"
        self.analytics_file = "feedback_analytics.json"
        self.embeddings_file = "embeddings.npy"
        self.faiss_file = "faiss_index.bin"
        
        self.load_database()
        self.load_feedback_data()
        self.load_embeddings()
        
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
        self.update_embeddings()
    
    def update_embeddings(self):
        """Update embeddings for semantic search"""
        if not self.embedding_model or not self.documents:
            return
        
        try:
            print(f"Generating embeddings for {len(self.documents)} documents...")
            self.embeddings = self.embedding_model.encode(self.documents)
            
            # Create FAISS index
            if faiss:
                dimension = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(self.embeddings.astype('float32'))
                
                # Save embeddings and index
                np.save(self.embeddings_file, self.embeddings)
                faiss.write_index(self.faiss_index, self.faiss_file)
                print("Embeddings and FAISS index updated")
        except Exception as e:
            print(f"Embedding update error: {e}")
    
    def load_embeddings(self):
        """Load existing embeddings and FAISS index"""
        try:
            if os.path.exists(self.embeddings_file) and os.path.exists(self.faiss_file):
                self.embeddings = np.load(self.embeddings_file)
                if faiss:
                    self.faiss_index = faiss.read_index(self.faiss_file)
                print(f"Loaded embeddings for {len(self.embeddings)} documents")
        except Exception as e:
            print(f"Load embeddings error: {e}")
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Perform semantic search using embeddings"""
        if not self.embedding_model or not self.faiss_index:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), n_results)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'score': float(score),
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                    })
            
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def hybrid_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Combine keyword and semantic search"""
        keyword_results = self._keyword_search(query, n_results)
        semantic_results = self.semantic_search(query, n_results)
        
        # Combine and deduplicate results
        combined = {}
        
        # Add keyword results with weight
        for result in keyword_results:
            doc_id = result.get('metadata', {}).get('id', result['text'][:50])
            combined[doc_id] = {
                **result,
                'combined_score': result['score'] * 0.6  # 60% weight for keyword
            }
        
        # Add semantic results with weight
        for result in semantic_results:
            doc_id = result.get('metadata', {}).get('id', result['text'][:50])
            if doc_id in combined:
                combined[doc_id]['combined_score'] += result['score'] * 0.4  # 40% weight for semantic
            else:
                combined[doc_id] = {
                    **result,
                    'combined_score': result['score'] * 0.4
                }
        
        # Sort by combined score
        results = list(combined.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results[:n_results]
    
    def analyze_medical_image(self, image_data: str, image_type: str, query: str = "", model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> Dict:
        """Analyze medical images (X-ray, MRI, CT, DICOM)"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Handle different image types
            if image_type.lower() == 'dicom':
                return self._analyze_dicom(image_bytes, query, image_data, model)
            else:
                return self._analyze_standard_image(image_bytes, image_type, query, image_data, model)
                
        except Exception as e:
            return {
                'analysis': f"Error analyzing image: {str(e)}",
                'findings': [],
                'recommendations': "Please ensure the image is properly formatted and try again.",
                'confidence': 0.0,
                'image_metadata': {}
            }
    
    def _analyze_dicom(self, image_bytes: bytes, query: str, image_base64: str = None, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> Dict:
        """Analyze DICOM medical images"""
        if not pydicom:
            return self._fallback_image_analysis("DICOM", query)
        
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(io.BytesIO(image_bytes))
            
            # Extract metadata
            metadata = {
                'modality': str(getattr(dicom_data, 'Modality', 'Unknown')),
                'body_part': str(getattr(dicom_data, 'BodyPartExamined', 'Unknown')),
                'study_date': str(getattr(dicom_data, 'StudyDate', 'Unknown')),
                'patient_age': str(getattr(dicom_data, 'PatientAge', 'Unknown')),
                'image_size': f"{dicom_data.Rows}x{dicom_data.Columns}" if hasattr(dicom_data, 'Rows') else 'Unknown'
            }
            
            # Convert DICOM to image for vision analysis
            try:
                # Convert DICOM to PIL Image
                pixel_array = dicom_data.pixel_array
                # Normalize to 0-255 range
                pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype('uint8')
                
                from PIL import Image
                img = Image.fromarray(pixel_array)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                dicom_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                
                # Generate AI analysis using Groq Vision
                analysis = self._generate_image_analysis(dicom_base64, metadata, query, 'DICOM')
            except Exception as e:
                analysis = f"DICOM processing error: {str(e)}. Manual review required."
            
            return {
                'analysis': analysis,
                'findings': self._extract_findings(analysis),
                'recommendations': self._generate_recommendations(metadata, analysis),
                'confidence': 0.8,
                'image_metadata': metadata
            }
            
        except Exception as e:
            return self._fallback_image_analysis("DICOM", query, str(e))
    
    def _analyze_standard_image(self, image_bytes: bytes, image_type: str, query: str, image_base64: str = None, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> Dict:
        """Analyze standard medical images (X-ray, MRI, CT)"""
        if not Image or not cv2:
            return self._fallback_image_analysis(image_type, query)
        
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image)
            
            # Enhanced image analysis
            metadata = {
                'format': image.format or 'Unknown',
                'dimensions': f"{image.width}x{image.height}",
                'color_mode': image.mode,
                'modality': image_type.upper(),
                'file_size_kb': len(image_bytes) / 1024
            }
            
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Enhanced image statistics
            metadata.update({
                'mean_intensity': float(np.mean(gray)),
                'intensity_range': f"{int(np.min(gray))}-{int(np.max(gray))}",
                'contrast_ratio': float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0,
                'histogram_peaks': len([i for i in range(1, len(np.histogram(gray, bins=50)[0])-1) 
                                      if np.histogram(gray, bins=50)[0][i] > np.histogram(gray, bins=50)[0][i-1] 
                                      and np.histogram(gray, bins=50)[0][i] > np.histogram(gray, bins=50)[0][i+1]])
            })
            
            # Add image quality assessment
            if metadata['contrast_ratio'] > 0.3:
                metadata['quality_assessment'] = 'Good contrast'
            elif metadata['contrast_ratio'] > 0.15:
                metadata['quality_assessment'] = 'Moderate contrast'
            else:
                metadata['quality_assessment'] = 'Low contrast - may affect diagnostic quality'
            
            # Generate AI analysis using Groq Vision
            analysis = self._generate_image_analysis(image_base64 or image_data, metadata, query, image_type, model)
            
            # Adjust confidence based on image quality
            base_confidence = 0.7
            if metadata.get('contrast_ratio', 0) < 0.15:
                base_confidence *= 0.6  # Low quality reduces confidence
            elif metadata.get('quality_assessment') == 'Good contrast':
                base_confidence *= 1.1  # Good quality increases confidence
            
            return {
                'analysis': analysis,
                'findings': self._extract_findings(analysis),
                'recommendations': self._generate_recommendations(metadata, analysis),
                'confidence': min(base_confidence, 0.85),  # Cap at 85%
                'image_metadata': metadata
            }
            
        except Exception as e:
            return self._fallback_image_analysis(image_type, query, str(e))
    
    def _generate_image_analysis(self, image_base64: str, metadata: Dict, query: str, image_type: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
        """Generate AI-powered image analysis using Groq Vision"""
        if not self.groq_client:
            return "AI analysis unavailable. Please consult with a radiologist for professional image interpretation."
        
        prompt = f"""You are an expert orthopedic radiologist. Analyze this {image_type.upper()} image systematically.

Clinical Question: {query if query else 'Provide systematic radiological interpretation'}

Provide analysis in this format:

**TECHNIQUE & QUALITY:**
- Image positioning, penetration, and diagnostic adequacy

**SYSTEMATIC REVIEW:**
- Bones: Examine cortical integrity, trabecular pattern, alignment
- Joints: Assess joint spaces, articular surfaces
- Soft tissues: Look for swelling, calcifications

**FINDINGS:**
- Describe what you actually see in the image
- Fractures: Specify location, pattern (transverse/oblique/spiral), displacement
- Degenerative changes: Joint space narrowing, osteophytes
- Other abnormalities if present

**IMPRESSION:**
- Primary findings based on visual analysis
- Clinical significance

**RECOMMENDATIONS:**
- Additional views or imaging if needed
- Clinical correlation requirements

Be precise and conservative. Only describe what is clearly visible. Avoid speculation.

**DISCLAIMER:** AI analysis for educational purposes only. Professional radiologist interpretation required."""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                model=model,
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to text-based analysis if vision fails
            if "invalid_api_key" in str(e).lower():
                return self._fallback_text_analysis(metadata, query, image_type)
            return f"Vision analysis error: {str(e)}. Please consult with a radiologist for professional interpretation."
    
    def _fallback_text_analysis(self, metadata: Dict, query: str, image_type: str) -> str:
        """Fallback analysis when vision API is unavailable"""
        return f"""**{image_type.upper()} IMAGE ANALYSIS**

**TECHNICAL INFORMATION:**
- Image Type: {image_type.upper()}
- Dimensions: {metadata.get('dimensions', 'Unknown')}
- Quality: {metadata.get('quality_assessment', 'Assessment unavailable')}

**ANALYSIS LIMITATION:**
Vision analysis currently unavailable. This is a technical analysis based on image metadata only.

**CLINICAL CORRELATION REQUIRED:**
For accurate radiological interpretation, please:
1. Ensure proper image quality and positioning
2. Consult with a qualified radiologist
3. Correlate with clinical history and examination

**NEXT STEPS:**
- Professional radiologist review required
- Clinical correlation with patient symptoms
- Consider additional imaging if clinically indicated

**DISCLAIMER:** This technical assessment cannot replace professional radiological interpretation."""
    
    def _extract_findings(self, analysis: str) -> List[str]:
        """Extract key findings from analysis text"""
        findings = []
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['finding', 'abnormal', 'concern', 'suggest', 'indicate']):
                if len(line) > 10 and not line.startswith('#'):
                    findings.append(line)
        
        return findings[:5]  # Return top 5 findings
    
    def _generate_recommendations(self, metadata: Dict, analysis: str) -> str:
        """Generate clinical recommendations based on analysis"""
        recommendations = []
        
        # Basic recommendations based on image type
        image_type = metadata.get('image_type', metadata.get('modality', 'Unknown')).upper()
        
        if 'X-RAY' in image_type or 'XR' in image_type:
            recommendations.append("Consider clinical correlation with patient symptoms")
            recommendations.append("Follow institutional protocols for X-ray interpretation")
        elif 'MRI' in image_type:
            recommendations.append("Correlate with clinical history and physical examination")
            recommendations.append("Consider additional MRI sequences if clinically indicated")
        elif 'CT' in image_type:
            recommendations.append("Correlate with clinical presentation")
            recommendations.append("Consider contrast studies if additional detail needed")
        
        recommendations.append("Professional radiologist review required for definitive diagnosis")
        
        return " • ".join(recommendations)
    
    def _fallback_image_analysis(self, image_type: str, query: str, error: str = "") -> Dict:
        """Fallback analysis when image processing libraries unavailable"""
        analysis = f"""Image Analysis for {image_type.upper()}:

This appears to be a {image_type} medical image. While I cannot perform detailed image processing analysis{' due to: ' + error if error else ''}, I can provide general guidance:

1. GENERAL APPROACH:
   - Systematic review of all visible structures
   - Comparison with normal anatomy
   - Assessment of image quality and technique

2. CLINICAL CORRELATION:
   - Always correlate imaging findings with clinical history
   - Consider patient symptoms and physical examination
   - Follow institutional imaging protocols

3. PROFESSIONAL REVIEW:
   - All medical images require professional radiologist interpretation
   - This AI analysis is for educational purposes only
   - Seek immediate medical attention for urgent clinical concerns

For detailed image analysis, please consult with a qualified radiologist."""
        
        return {
            'analysis': analysis,
            'findings': [f"Professional {image_type} interpretation required"],
            'recommendations': "Consult with radiologist for professional image interpretation",
            'confidence': 0.3,
            'image_metadata': {'type': image_type, 'status': 'Limited analysis available'}
        }
    
    def search_documents(self, query: str, n_results: int = 5, use_hybrid: bool = True) -> List[Dict]:
        """Enhanced search with hybrid semantic + keyword matching"""
        if use_hybrid and self.embedding_model:
            return self.hybrid_search(query, n_results)
        
        return self._keyword_search(query, n_results)
    
    def _keyword_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Pure keyword search implementation"""
        query = query.lower().strip()
        results = []
        
        # Enhanced keyword matching with partial matches
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            score = 0
            
            # Count exact keyword matches
            for word in query.split():
                if len(word) > 2:  # Only search meaningful words
                    # Exact match
                    if word in doc_lower:
                        score += doc_lower.count(word) * 2
                    # Partial match (word contains or is contained)
                    else:
                        for doc_word in doc_lower.split():
                            if word in doc_word or doc_word in word:
                                if len(doc_word) > 3:  # Avoid short word false matches
                                    score += 0.5
            
            if score > 0:
                results.append({
                    'text': doc,
                    'score': score / len([w for w in query.split() if len(w) > 2]),
                    'metadata': self.metadata[i] if i < len(self.metadata) else {}
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    def generate_response(self, query: str, context_results: List[Dict], chat_history: List[Dict] = None, mode: str = "normal", model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
        if not self.groq_client:
            return "AI service unavailable. Please try again later."
            
        return self._generate_complete_response(query, context_results, chat_history, mode, model)
    
    def _generate_complete_response(self, query: str, context_results: List[Dict], chat_history: List[Dict] = None, mode: str = "normal", model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
        """Generate BoneQuest orthopedic AI response with role-based formatting"""
        if not context_results:
            if mode == "normal":
                return "I don't have enough information to answer your question safely. Please consult with a healthcare professional for proper medical advice."
            else:
                return "Insufficient evidence in knowledge base. Recommend consulting primary literature or clinical guidelines."
        
        # Build evidence context with citations
        evidence_parts = []
        for i, result in enumerate(context_results[:3]):
            evidence_parts.append(f"[{i+1}] {result['text'][:800]}")
        
        evidence_text = "\n\n".join(evidence_parts)
        
        # Get feedback mode from environment (set by admin)
        feedback_mode = os.getenv("FEEDBACK_MODE", "OFF")
        
        if mode == "normal":
            prompt = f"""You are BoneQuest — Patient Mode.

Goal: Explain orthopedic topics to patients and non-specialists in clear, empathetic, low-jargon language.

Evidence from medical documents:
{evidence_text}

User Question: {query}

Behavior:
- Use plain language and short sentences. If you must use a medical term, follow it immediately with a simple explanation or analogy.
- Start with a 1–2 line summary (TL;DR), then a short explanation, then 1–2 practical next steps.
- Use a reassuring, respectful tone. Avoid alarming language.
- When uncertain, say "I don't know" or "I can't be sure from this information" and recommend seeing a clinician.
- Do NOT provide definitive diagnoses or prescriptive surgical instructions. Always include: "This is educational information and not a substitute for professional medical advice."

Formatting:
- TL;DR:
- What it might mean:
- Simple explanation:
- Next steps / When to seek care:
- Confidence: (low/med/high)

Provide response:"""
        elif mode == "jr":
            prompt = f"""You are BoneQuest — Junior Resident Mode.

Goal: Provide educational, learning-focused responses for junior residents with basic clinical knowledge.

Evidence from medical documents:
{evidence_text}

User Question: {query}

Behavior:
- Use medical terminology with brief explanations
- Focus on learning objectives and key concepts
- Include differential diagnosis with reasoning
- Provide step-by-step clinical approach
- Mention common pitfalls and red flags
- Reference basic anatomy and pathophysiology

Formatting:
- Clinical Summary:
- Key Learning Points:
- Differential Diagnosis:
- Clinical Approach:
- Red Flags:
- Further Reading: [cite sources]

Provide educational response:"""
        elif mode == "sr":
            prompt = f"""You are BoneQuest — Senior Resident Mode.

Goal: Provide advanced clinical guidance for senior residents with intermediate expertise.

Evidence from medical documents:
{evidence_text}

User Question: {query}

Behavior:
- Use advanced medical terminology
- Focus on clinical decision-making and management
- Include evidence-based treatment options
- Discuss surgical vs conservative management
- Mention complications and their management
- Reference current guidelines and protocols

Formatting:
- Clinical Assessment:
- Management Options:
- Surgical Considerations:
- Complications:
- Evidence Level:
- Guidelines: [cite sources]

Provide clinical guidance:"""
        else:  # consultant mode
            prompt = f"""You are BoneQuest — Consultant Mode.

Goal: Provide expert-level clinical consultation for consultants and specialists.

Evidence from medical documents:
{evidence_text}

User Question: {query}

Behavior:
- Use precise medical/technical language and cite anatomy, imaging findings, or guideline references
- Focus on complex cases, latest research, and expert opinion
- Include detailed surgical techniques and advanced management
- Discuss controversial topics and emerging evidence
- Provide teaching points for trainees
- Reference high-level evidence and recent literature

Formatting:
- Expert Opinion:
- Advanced Management:
- Surgical Technique:
- Current Evidence:
- Teaching Points:
- References: [cite sources]

Provide expert consultation:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.2,
                max_tokens=600
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Check for corrupted response
            if not self._is_response_complete(response_text):
                return "I'm experiencing technical difficulties generating a complete response. Please try rephrasing your question."
            
            # Only add feedback prompt if enabled
            if feedback_mode == "ON":
                response_text += "\n\nWas this answer helpful? (👍/👎)"
            
            return response_text
            
        except Exception as e:
            return f"I'm experiencing technical difficulties. Please try again. Error: {str(e)[:100]}"
    
    def _is_response_complete(self, response: str) -> bool:
        """Check if response appears complete"""
        if not response or len(response) < 10:
            return False
        
        response = response.strip()
        
        # Check for corrupted responses (repeated numbers/characters)
        if len(set(response[-20:])) < 3:  # Too many repeated characters
            return False
        
        # Check for proper sentence ending
        if response.endswith(('.', '!', '?')):
            return True
        
        # Check if response seems truncated
        if len(response) > 100 and not response[-1].isspace():
            return True
        
        return len(response) > 50
    
    def generate_speech_response(self, query: str, context_results: List[Dict]) -> str:
        if not self.groq_client:
            return "AI service unavailable."
            
        if not context_results:
            prompt = f"""You are BoneQuest orthopedic AI. Answer briefly in 1-2 sentences: "{query}"

Provide a clear, direct medical response for speech delivery."""
            
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
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
    
    def save_feedback(self, feedback: Dict):
        """Save user feedback for analysis and improvement"""
        try:
            feedback_data = self.load_feedback_data()
            feedback['id'] = f"fb_{len(feedback_data)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            feedback['timestamp'] = datetime.now().isoformat()
            feedback_data.append(feedback)
            
            with open(self.feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            # Update analytics
            self.update_feedback_analytics(feedback_data)
            print(f"Feedback saved: {feedback['feedback_type']} rating: {feedback['rating']}")
            return True
        except Exception as e:
            print(f"Save feedback error: {e}")
            return False
    
    def load_feedback_data(self) -> List[Dict]:
        """Load all feedback data"""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Load feedback error: {e}")
                return []
        return []
    
    def update_feedback_analytics(self, feedback_data: List[Dict]):
        """Update feedback analytics and insights"""
        try:
            if not feedback_data:
                return
            
            total_feedback = len(feedback_data)
            ratings = [fb['rating'] for fb in feedback_data if fb.get('rating')]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            positive_count = len([fb for fb in feedback_data if fb.get('feedback_type') == 'positive'])
            negative_count = len([fb for fb in feedback_data if fb.get('feedback_type') == 'negative'])
            correction_count = len([fb for fb in feedback_data if fb.get('feedback_type') == 'correction'])
            
            # Analyze common issues
            common_issues = self.analyze_common_issues(feedback_data)
            improvement_suggestions = self.generate_improvement_suggestions(feedback_data)
            
            analytics = {
                'total_feedback': total_feedback,
                'avg_rating': round(avg_rating, 2),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'correction_count': correction_count,
                'common_issues': common_issues,
                'improvement_suggestions': improvement_suggestions,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.analytics_file, 'w') as f:
                json.dump(analytics, f, indent=2)
                
        except Exception as e:
            print(f"Update analytics error: {e}")
    
    def analyze_common_issues(self, feedback_data: List[Dict]) -> List[Dict]:
        """Analyze feedback to identify common issues"""
        issues = {}
        
        for fb in feedback_data:
            if fb.get('feedback_type') == 'negative' or fb.get('rating', 5) < 3:
                comment = fb.get('comment', '').lower()
                
                # Categorize issues
                if 'accuracy' in comment or 'wrong' in comment or 'incorrect' in comment:
                    issues['accuracy'] = issues.get('accuracy', 0) + 1
                elif 'incomplete' in comment or 'missing' in comment or 'more detail' in comment:
                    issues['completeness'] = issues.get('completeness', 0) + 1
                elif 'search' in comment or 'relevant' in comment or 'context' in comment:
                    issues['search_quality'] = issues.get('search_quality', 0) + 1
                elif 'slow' in comment or 'speed' in comment or 'time' in comment:
                    issues['performance'] = issues.get('performance', 0) + 1
                else:
                    issues['other'] = issues.get('other', 0) + 1
        
        # Convert to list format
        return [{'issue': k, 'count': v} for k, v in sorted(issues.items(), key=lambda x: x[1], reverse=True)]
    
    def generate_improvement_suggestions(self, feedback_data: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on feedback"""
        suggestions = []
        
        # Analyze ratings by category
        context_ratings = [fb.get('context_quality', 3) for fb in feedback_data]
        accuracy_ratings = [fb.get('response_accuracy', 3) for fb in feedback_data]
        completeness_ratings = [fb.get('response_completeness', 3) for fb in feedback_data]
        search_ratings = [fb.get('search_relevance', 3) for fb in feedback_data]
        
        if context_ratings and sum(context_ratings) / len(context_ratings) < 3:
            suggestions.append("Improve document chunking and context extraction")
        
        if accuracy_ratings and sum(accuracy_ratings) / len(accuracy_ratings) < 3:
            suggestions.append("Enhance fact-checking and evidence validation")
        
        if completeness_ratings and sum(completeness_ratings) / len(completeness_ratings) < 3:
            suggestions.append("Provide more comprehensive responses")
        
        if search_ratings and sum(search_ratings) / len(search_ratings) < 3:
            suggestions.append("Optimize search algorithm and relevance scoring")
        
        # Check for correction patterns
        corrections = [fb.get('correction', '') for fb in feedback_data if fb.get('correction')]
        if len(corrections) > 5:
            suggestions.append("Review and incorporate user corrections into knowledge base")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def get_feedback_analytics(self) -> Dict:
        """Get current feedback analytics"""
        if os.path.exists(self.analytics_file):
            try:
                with open(self.analytics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Load analytics error: {e}")
        
        return {
            'total_feedback': 0,
            'avg_rating': 0,
            'positive_count': 0,
            'negative_count': 0,
            'correction_count': 0,
            'common_issues': [],
            'improvement_suggestions': []
        }
    
    def get_search_insights(self, query: str) -> Dict:
        """Analyze search performance for a query"""
        feedback_data = self.load_feedback_data()
        query_feedback = [fb for fb in feedback_data if query.lower() in fb.get('query', '').lower()]
        
        if not query_feedback:
            return {'message': 'No feedback available for similar queries'}
        
        avg_search_rating = sum([fb.get('search_relevance', 3) for fb in query_feedback]) / len(query_feedback)
        avg_accuracy = sum([fb.get('response_accuracy', 3) for fb in query_feedback]) / len(query_feedback)
        
        return {
            'similar_queries': len(query_feedback),
            'avg_search_rating': round(avg_search_rating, 2),
            'avg_accuracy': round(avg_accuracy, 2),
            'needs_improvement': avg_search_rating < 3 or avg_accuracy < 3
        }

rag_system = AdvancedBoneQuestRAG()

@app.get("/")
async def read_root():
    return FileResponse('frontend/index.html')

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    context_results = rag_system.search_documents(request.query)
    response = rag_system.generate_response(request.query, context_results, request.chat_history, request.mode, request.model)
    
    # Add RAG indicator
    rag_sources = len(context_results)
    if rag_sources > 0:
        response += f"\n\n📚 *Response based on {rag_sources} knowledge base sources*"
    else:
        response += "\n\n🤖 *Response from general AI knowledge (no RAG sources found)*"
    
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

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    sessions = rag_system.load_sessions()
    if session_id in sessions:
        return sessions[session_id]
    else:
        raise HTTPException(status_code=404, detail="Session not found")

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

class FeedbackModeRequest(BaseModel):
    enabled: bool

@app.post("/admin/feedback-mode")
async def toggle_feedback_mode(request: FeedbackModeRequest):
    os.environ["FEEDBACK_MODE"] = "ON" if request.enabled else "OFF"
    return {"feedback_mode": os.environ["FEEDBACK_MODE"]}

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit detailed user feedback"""
    feedback_data = {
        'message_id': request.message_id,
        'query': request.query,
        'response': request.response,
        'feedback_type': request.feedback_type,
        'rating': request.rating,
        'comment': request.comment,
        'correction': request.correction,
        'context_quality': request.context_quality,
        'response_accuracy': request.response_accuracy,
        'response_completeness': request.response_completeness,
        'search_relevance': request.search_relevance
    }
    
    success = rag_system.save_feedback(feedback_data)
    if success:
        return {"message": "Feedback submitted successfully", "status": "success"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@app.get("/feedback/analytics")
async def get_feedback_analytics():
    """Get comprehensive feedback analytics"""
    return rag_system.get_feedback_analytics()

@app.get("/feedback/insights/{query}")
async def get_search_insights(query: str):
    """Get search performance insights for a specific query"""
    return rag_system.get_search_insights(query)

@app.get("/feedback/data")
async def get_feedback_data():
    """Get all feedback data (admin only)"""
    return rag_system.load_feedback_data()

@app.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze medical images (X-ray, MRI, CT, DICOM)"""
    try:
        result = rag_system.analyze_medical_image(
            request.image_data, 
            request.image_type, 
            request.query,
            request.model
        )
        
        return ImageAnalysisResponse(
            analysis=result['analysis'],
            findings=result['findings'],
            recommendations=result['recommendations'],
            confidence=result['confidence'],
            image_metadata=result['image_metadata']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...), image_type: str = "xray"):
    """Upload and analyze medical image file"""
    allowed_types = ["image/jpeg", "image/png", "image/tiff", "application/dicom"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    try:
        image_bytes = await file.read()
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        result = rag_system.analyze_medical_image(image_data, image_type)
        
        return {
            "filename": file.filename,
            "analysis": result['analysis'],
            "findings": result['findings'],
            "recommendations": result['recommendations'],
            "confidence": result['confidence'],
            "metadata": result['image_metadata']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

@app.get("/search-methods")
async def get_search_methods():
    """Get available search methods"""
    return {
        "semantic_search": rag_system.embedding_model is not None,
        "hybrid_search": rag_system.embedding_model is not None and rag_system.faiss_index is not None,
        "keyword_search": True,
        "image_analysis": True
    }

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    return {"message": "DevTools configuration not available"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "groq_available": rag_system.groq_client is not None,
        "documents_loaded": len(rag_system.documents),
        "feedback_mode": os.getenv("FEEDBACK_MODE", "OFF"),
        "semantic_search": rag_system.embedding_model is not None,
        "image_analysis": Image is not None and cv2 is not None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)