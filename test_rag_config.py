#!/usr/bin/env python3
"""
Advanced RAG Configuration Test
Tests all RAG components to verify proper setup
"""

import requests
import json
import sys
import os

class RAGConfigTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
    
    def test_server_health(self):
        """Test if server is running and healthy"""
        print("🔍 Testing server health...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.results['server'] = {
                    'status': 'healthy',
                    'groq_available': data.get('groq_available', False),
                    'documents_loaded': data.get('documents_loaded', 0),
                    'semantic_search': data.get('semantic_search', False),
                    'image_analysis': data.get('image_analysis', False)
                }
                print("✅ Server is healthy")
                return True
            else:
                print(f"❌ Server unhealthy: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Server connection failed: {e}")
            self.results['server'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_search_methods(self):
        """Test available search methods"""
        print("\n🔍 Testing search methods...")
        try:
            response = requests.get(f"{self.base_url}/search-methods", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.results['search_methods'] = data
                
                print(f"  Semantic Search: {'✅' if data.get('semantic_search') else '❌'}")
                print(f"  Hybrid Search: {'✅' if data.get('hybrid_search') else '❌'}")
                print(f"  Keyword Search: {'✅' if data.get('keyword_search') else '❌'}")
                print(f"  Image Analysis: {'✅' if data.get('image_analysis') else '❌'}")
                return True
            else:
                print(f"❌ Search methods test failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Search methods test failed: {e}")
            return False
    
    def test_document_stats(self):
        """Test document statistics"""
        print("\n🔍 Testing document statistics...")
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.results['stats'] = data
                
                doc_count = data.get('document_count', 0)
                sources = data.get('sources', 0)
                
                print(f"  Documents loaded: {doc_count}")
                print(f"  Unique sources: {sources}")
                
                if doc_count > 0:
                    print("✅ Knowledge base has documents")
                    return True
                else:
                    print("⚠️  No documents in knowledge base")
                    return False
            else:
                print(f"❌ Stats test failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Stats test failed: {e}")
            return False
    
    def test_basic_chat(self):
        """Test basic chat functionality"""
        print("\n🔍 Testing basic chat...")
        test_query = "What is orthopedics?"
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "query": test_query,
                    "mode": "normal",
                    "model": "llama-3.1-8b-instant"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')
                context = data.get('context', [])
                
                self.results['basic_chat'] = {
                    'success': True,
                    'response_length': len(response_text),
                    'context_sources': len(context),
                    'has_rag_indicator': '📚' in response_text or '🤖' in response_text
                }
                
                print(f"  Response length: {len(response_text)} chars")
                print(f"  Context sources: {len(context)}")
                print(f"  RAG indicator: {'✅' if '📚' in response_text or '🤖' in response_text else '❌'}")
                
                if len(context) > 0:
                    print("✅ RAG system found relevant documents")
                else:
                    print("⚠️  No RAG context found (using general AI)")
                
                return True
            else:
                print(f"❌ Chat test failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Chat test failed: {e}")
            return False
    
    def test_role_modes(self):
        """Test different role modes"""
        print("\n🔍 Testing role modes...")
        roles = ["normal", "jr", "sr", "consultant"]
        test_query = "How do you treat a fracture?"
        
        role_results = {}
        
        for role in roles:
            try:
                response = requests.post(
                    f"{self.base_url}/chat",
                    json={
                        "query": test_query,
                        "mode": role,
                        "model": "llama-3.1-8b-instant"
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    context = data.get('context', [])
                    
                    role_results[role] = {
                        'success': True,
                        'response_length': len(response_text),
                        'context_sources': len(context)
                    }
                    print(f"  {role}: ✅ ({len(response_text)} chars, {len(context)} sources)")
                else:
                    role_results[role] = {'success': False, 'error': f"HTTP {response.status_code}"}
                    print(f"  {role}: ❌ HTTP {response.status_code}")
            except Exception as e:
                role_results[role] = {'success': False, 'error': str(e)}
                print(f"  {role}: ❌ {e}")
        
        self.results['role_modes'] = role_results
        successful_roles = sum(1 for r in role_results.values() if r.get('success'))
        return successful_roles == len(roles)
    
    def test_ai_models(self):
        """Test different AI models"""
        print("\n🔍 Testing AI models...")
        models = [
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b"
        ]
        test_query = "What is a bone fracture?"
        
        model_results = {}
        
        for model in models:
            try:
                response = requests.post(
                    f"{self.base_url}/chat",
                    json={
                        "query": test_query,
                        "mode": "normal",
                        "model": model
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    
                    model_results[model] = {
                        'success': True,
                        'response_length': len(response_text)
                    }
                    print(f"  {model}: ✅ ({len(response_text)} chars)")
                else:
                    model_results[model] = {'success': False, 'error': f"HTTP {response.status_code}"}
                    print(f"  {model}: ❌ HTTP {response.status_code}")
            except Exception as e:
                model_results[model] = {'success': False, 'error': str(e)}
                print(f"  {model}: ❌ {e}")
        
        self.results['ai_models'] = model_results
        successful_models = sum(1 for r in model_results.values() if r.get('success'))
        return successful_models > 0
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("🎯 ADVANCED RAG CONFIGURATION TEST REPORT")
        print("="*60)
        
        # Server Health
        server_status = self.results.get('server', {})
        print(f"\n🖥️  SERVER STATUS:")
        print(f"  Status: {server_status.get('status', 'unknown')}")
        print(f"  Groq API: {'✅' if server_status.get('groq_available') else '❌'}")
        print(f"  Documents: {server_status.get('documents_loaded', 0)}")
        print(f"  Semantic Search: {'✅' if server_status.get('semantic_search') else '❌'}")
        
        # Search Methods
        search_methods = self.results.get('search_methods', {})
        print(f"\n🔍 SEARCH CAPABILITIES:")
        print(f"  Semantic Search: {'✅' if search_methods.get('semantic_search') else '❌'}")
        print(f"  Hybrid Search: {'✅' if search_methods.get('hybrid_search') else '❌'}")
        print(f"  Keyword Search: {'✅' if search_methods.get('keyword_search') else '❌'}")
        
        # Knowledge Base
        stats = self.results.get('stats', {})
        print(f"\n📚 KNOWLEDGE BASE:")
        print(f"  Documents: {stats.get('document_count', 0)}")
        print(f"  Sources: {stats.get('sources', 0)}")
        
        # Chat Functionality
        chat = self.results.get('basic_chat', {})
        print(f"\n💬 CHAT FUNCTIONALITY:")
        print(f"  Basic Chat: {'✅' if chat.get('success') else '❌'}")
        print(f"  RAG Indicator: {'✅' if chat.get('has_rag_indicator') else '❌'}")
        print(f"  Context Sources: {chat.get('context_sources', 0)}")
        
        # Role Modes
        roles = self.results.get('role_modes', {})
        print(f"\n👥 ROLE MODES:")
        for role, result in roles.items():
            status = '✅' if result.get('success') else '❌'
            print(f"  {role}: {status}")
        
        # AI Models
        models = self.results.get('ai_models', {})
        print(f"\n🤖 AI MODELS:")
        for model, result in models.items():
            status = '✅' if result.get('success') else '❌'
            print(f"  {model.split('/')[-1]}: {status}")
        
        # Overall Assessment
        print(f"\n📊 OVERALL ASSESSMENT:")
        
        # Calculate scores
        server_score = 1 if server_status.get('status') == 'healthy' else 0
        groq_score = 1 if server_status.get('groq_available') else 0
        docs_score = 1 if stats.get('document_count', 0) > 0 else 0
        semantic_score = 1 if search_methods.get('semantic_search') else 0
        chat_score = 1 if chat.get('success') else 0
        
        total_score = server_score + groq_score + docs_score + semantic_score + chat_score
        max_score = 5
        
        percentage = (total_score / max_score) * 100
        
        if percentage >= 80:
            grade = "A - Excellent"
            status = "🟢 Fully Operational"
        elif percentage >= 60:
            grade = "B - Good"
            status = "🟡 Mostly Operational"
        elif percentage >= 40:
            grade = "C - Fair"
            status = "🟠 Partially Operational"
        else:
            grade = "D - Poor"
            status = "🔴 Needs Attention"
        
        print(f"  Configuration Score: {total_score}/{max_score} ({percentage:.0f}%)")
        print(f"  Grade: {grade}")
        print(f"  Status: {status}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not server_status.get('groq_available'):
            print("  - Check GROQ_API_KEY environment variable")
        if stats.get('document_count', 0) == 0:
            print("  - Upload documents via Admin panel to enable RAG")
        if not search_methods.get('semantic_search'):
            print("  - Install sentence-transformers for semantic search")
        if not chat.get('success'):
            print("  - Check server logs for chat errors")
        
        print("\n" + "="*60)

def main():
    """Run comprehensive RAG configuration test"""
    print("🚀 Advanced RAG Configuration Test")
    print("Testing all components of your BoneQuest RAG system...")
    
    tester = RAGConfigTester()
    
    # Run all tests
    tests = [
        tester.test_server_health,
        tester.test_search_methods,
        tester.test_document_stats,
        tester.test_basic_chat,
        tester.test_role_modes,
        tester.test_ai_models
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Generate report
    tester.generate_report()
    
    # Save results
    with open("rag_config_test.json", "w") as f:
        json.dump(tester.results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: rag_config_test.json")

if __name__ == "__main__":
    main()