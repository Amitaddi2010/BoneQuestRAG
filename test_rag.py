#!/usr/bin/env python3
"""
RAG System Test Script for BoneQuest
Tests document processing, vectorization, and retrieval functionality
"""

import os
import sys
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

def test_rag_system():
    print("🧪 Testing BoneQuest RAG System")
    print("=" * 50)
    
    # Test documents
    test_docs = [
        "Neuromuscular deformities are conditions affecting muscles and nerves. Common types include cerebral palsy, muscular dystrophy, and spina bifida. These conditions can cause joint contractures, muscle weakness, and skeletal deformities.",
        "Treatment for neuromuscular deformities often involves physical therapy, occupational therapy, and sometimes surgical intervention. Orthotic devices and mobility aids are commonly used to improve function.",
        "Spinal deformities such as scoliosis and kyphosis are common in neuromuscular conditions. Early detection and treatment are crucial for preventing progression and maintaining quality of life.",
        "Machine learning algorithms are used in artificial intelligence applications. Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Python programming language is widely used for data science and web development. It has extensive libraries for scientific computing and machine learning."
    ]
    
    # Initialize vectorizer with same parameters as BoneQuest
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True,
        strip_accents='ascii'
    )
    
    print("📄 Processing test documents...")
    document_vectors = vectorizer.fit_transform(test_docs)
    print(f"✅ Processed {len(test_docs)} documents")
    print(f"📊 Vector shape: {document_vectors.shape}")
    
    # Test queries
    test_queries = [
        "What are neuromuscular deformities?",
        "How to treat spinal deformities?",
        "Tell me about machine learning",
        "What is cerebral palsy?",
        "Explain scoliosis treatment"
    ]
    
    print("\n🔍 Testing retrieval for different queries:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        # Vectorize query
        query_vector = vectorizer.transform([query.lower().strip()])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        
        # Get top 3 results
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        print("📊 Results:")
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            doc_preview = test_docs[idx][:100] + "..." if len(test_docs[idx]) > 100 else test_docs[idx]
            print(f"  {i+1}. Score: {score:.4f} | Doc: {doc_preview}")
        
        # Check if retrieval is working
        best_score = similarities[top_indices[0]]
        if best_score > 0.1:
            print(f"  ✅ Good retrieval (score: {best_score:.4f})")
        elif best_score > 0.05:
            print(f"  ⚠️  Moderate retrieval (score: {best_score:.4f})")
        else:
            print(f"  ❌ Poor retrieval (score: {best_score:.4f})")
    
    print("\n" + "=" * 50)
    print("🎯 RAG System Analysis:")
    
    # Test specific medical query
    medical_query = "neuromuscular deformity treatment"
    query_vector = vectorizer.transform([medical_query])
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    medical_scores = similarities[:3]  # First 3 docs are medical
    non_medical_scores = similarities[3:]  # Last 2 are non-medical
    
    print(f"📋 Medical query: '{medical_query}'")
    print(f"   Medical docs avg score: {np.mean(medical_scores):.4f}")
    print(f"   Non-medical docs avg score: {np.mean(non_medical_scores):.4f}")
    
    if np.mean(medical_scores) > np.mean(non_medical_scores):
        print("   ✅ RAG correctly prioritizes relevant content")
    else:
        print("   ❌ RAG may have relevance issues")
    
    # Vocabulary analysis
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n📚 Vocabulary size: {len(feature_names)}")
    print(f"   Sample terms: {list(feature_names[:10])}")
    
    # Check for medical terms
    medical_terms = ['neuromuscular', 'deformity', 'therapy', 'treatment', 'spinal']
    found_terms = [term for term in medical_terms if term in feature_names]
    print(f"   Medical terms found: {found_terms}")
    
    print("\n🏁 Test completed!")
    return True

if __name__ == "__main__":
    try:
        test_rag_system()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)