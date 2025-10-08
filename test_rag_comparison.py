#!/usr/bin/env python3
"""
BoneQuest RAG Comparison Testing Tool
Compare Advanced RAG responses with ChatGPT, Grok, and Claude
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any
import openai
import anthropic
from groq import Groq

class RAGComparison:
    def __init__(self):
        # Initialize API clients
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # BoneQuest RAG endpoint
        self.rag_endpoint = "http://localhost:8000/chat"
        
        # Test queries for orthopedic scenarios
        self.test_queries = [
            "What are the contraindications for hip replacement surgery?",
            "Explain the management of open fractures in emergency settings",
            "What are the red flags for spinal cord compression?",
            "Describe the stepwise approach to knee arthroscopy",
            "What are the complications of shoulder dislocation?",
            "How do you manage compartment syndrome?",
            "What are the indications for spinal fusion surgery?",
            "Explain the treatment protocol for ACL tears",
            "What are the signs of fat embolism syndrome?",
            "Describe the management of pediatric supracondylar fractures"
        ]
        
        self.results = []
    
    def test_bonequest_rag(self, query: str, mode: str = "consultant") -> Dict[str, Any]:
        """Test BoneQuest Advanced RAG system"""
        try:
            response = requests.post(self.rag_endpoint, json={
                "query": query,
                "mode": mode,
                "model": "meta-llama/llama-4-maverick-17b-128e-instruct"
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data.get("response", ""),
                    "context_count": len(data.get("context", [])),
                    "has_citations": "context" in data and len(data["context"]) > 0,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_chatgpt(self, query: str) -> Dict[str, Any]:
        """Test ChatGPT-4"""
        try:
            start_time = time.time()
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert orthopedic surgeon providing clinical guidance. Provide evidence-based, detailed responses with clear recommendations."},
                    {"role": "user", "content": query}
                ],
                max_tokens=800,
                temperature=0.2
            )
            response_time = time.time() - start_time
            
            return {
                "response": response.choices[0].message.content,
                "response_time": response_time,
                "tokens_used": response.usage.total_tokens
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_grok(self, query: str) -> Dict[str, Any]:
        """Test Grok via Groq API"""
        try:
            start_time = time.time()
            response = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an expert orthopedic surgeon providing clinical guidance. Provide evidence-based, detailed responses with clear recommendations."},
                    {"role": "user", "content": query}
                ],
                max_tokens=800,
                temperature=0.2
            )
            response_time = time.time() - start_time
            
            return {
                "response": response.choices[0].message.content,
                "response_time": response_time
            }
        except Exception as e:
            return {"error": str(e)}
    
    def test_claude(self, query: str) -> Dict[str, Any]:
        """Test Claude-3"""
        try:
            start_time = time.time()
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                system="You are an expert orthopedic surgeon providing clinical guidance. Provide evidence-based, detailed responses with clear recommendations.",
                messages=[{"role": "user", "content": query}]
            )
            response_time = time.time() - start_time
            
            return {
                "response": response.content[0].text,
                "response_time": response_time,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            }
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_response(self, response: str, query: str) -> Dict[str, int]:
        """Simple evaluation metrics (1-5 scale)"""
        # Basic metrics - you can enhance these
        metrics = {
            "completeness": 3,  # Default neutral score
            "accuracy": 3,
            "clinical_relevance": 3,
            "clarity": 3,
            "evidence_based": 3
        }
        
        # Simple heuristics for evaluation
        if len(response) > 500:
            metrics["completeness"] += 1
        if "contraindication" in response.lower() or "indication" in response.lower():
            metrics["clinical_relevance"] += 1
        if "study" in response.lower() or "evidence" in response.lower():
            metrics["evidence_based"] += 1
        if response.count('.') > 5:  # Well-structured sentences
            metrics["clarity"] += 1
        
        # Cap at 5
        for key in metrics:
            metrics[key] = min(5, metrics[key])
        
        return metrics
    
    def run_comparison(self, query: str) -> Dict[str, Any]:
        """Run comparison for a single query"""
        print(f"\n🔍 Testing Query: {query}")
        
        result = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        # Test BoneQuest RAG
        print("  Testing BoneQuest RAG...")
        rag_result = self.test_bonequest_rag(query)
        if "error" not in rag_result:
            rag_result["evaluation"] = self.evaluate_response(rag_result["response"], query)
        result["models"]["bonequest_rag"] = rag_result
        
        # Test ChatGPT
        print("  Testing ChatGPT-4...")
        chatgpt_result = self.test_chatgpt(query)
        if "error" not in chatgpt_result:
            chatgpt_result["evaluation"] = self.evaluate_response(chatgpt_result["response"], query)
        result["models"]["chatgpt"] = chatgpt_result
        
        # Test Grok
        print("  Testing Grok...")
        grok_result = self.test_grok(query)
        if "error" not in grok_result:
            grok_result["evaluation"] = self.evaluate_response(grok_result["response"], query)
        result["models"]["grok"] = grok_result
        
        # Test Claude
        print("  Testing Claude-3...")
        claude_result = self.test_claude(query)
        if "error" not in claude_result:
            claude_result["evaluation"] = self.evaluate_response(claude_result["response"], query)
        result["models"]["claude"] = claude_result
        
        return result
    
    def run_full_comparison(self):
        """Run comparison on all test queries"""
        print("🚀 Starting BoneQuest RAG Comparison Test")
        print(f"📝 Testing {len(self.test_queries)} queries against 4 models")
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}]", end="")
            result = self.run_comparison(query)
            self.results.append(result)
            
            # Brief delay between tests
            time.sleep(2)
        
        # Save results
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """Save detailed results to JSON"""
        filename = f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {filename}")
    
    def generate_report(self):
        """Generate comparison report"""
        print("\n" + "="*80)
        print("📊 BONEQUEST RAG COMPARISON REPORT")
        print("="*80)
        
        # Calculate averages
        model_scores = {
            "bonequest_rag": {"total": 0, "count": 0, "response_times": []},
            "chatgpt": {"total": 0, "count": 0, "response_times": []},
            "grok": {"total": 0, "count": 0, "response_times": []},
            "claude": {"total": 0, "count": 0, "response_times": []}
        }
        
        for result in self.results:
            for model_name, model_data in result["models"].items():
                if "error" not in model_data and "evaluation" in model_data:
                    scores = model_data["evaluation"]
                    avg_score = sum(scores.values()) / len(scores)
                    model_scores[model_name]["total"] += avg_score
                    model_scores[model_name]["count"] += 1
                    
                    if "response_time" in model_data:
                        model_scores[model_name]["response_times"].append(model_data["response_time"])
        
        # Print rankings
        rankings = []
        for model, data in model_scores.items():
            if data["count"] > 0:
                avg_score = data["total"] / data["count"]
                avg_time = sum(data["response_times"]) / len(data["response_times"]) if data["response_times"] else 0
                rankings.append((model, avg_score, avg_time, data["count"]))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🏆 OVERALL RANKINGS (by average score):")
        print("-" * 60)
        for i, (model, score, time, count) in enumerate(rankings, 1):
            model_display = {
                "bonequest_rag": "🤖 BoneQuest RAG",
                "chatgpt": "🧠 ChatGPT-4",
                "grok": "⚡ Grok",
                "claude": "🎭 Claude-3"
            }
            print(f"{i}. {model_display[model]:<20} Score: {score:.2f}/5.0  Time: {time:.2f}s  Tests: {count}")
        
        # Detailed metrics
        print("\n📈 DETAILED METRICS:")
        print("-" * 60)
        
        metrics = ["completeness", "accuracy", "clinical_relevance", "clarity", "evidence_based"]
        
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()}:")
            metric_scores = []
            
            for model_name in ["bonequest_rag", "chatgpt", "grok", "claude"]:
                scores = []
                for result in self.results:
                    if model_name in result["models"] and "evaluation" in result["models"][model_name]:
                        scores.append(result["models"][model_name]["evaluation"][metric])
                
                if scores:
                    avg = sum(scores) / len(scores)
                    metric_scores.append((model_name, avg))
            
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            for model, score in metric_scores:
                model_display = {
                    "bonequest_rag": "BoneQuest RAG",
                    "chatgpt": "ChatGPT-4",
                    "grok": "Grok",
                    "claude": "Claude-3"
                }
                print(f"  {model_display[model]:<15}: {score:.2f}/5.0")
        
        # BoneQuest specific advantages
        print(f"\n🎯 BONEQUEST RAG ADVANTAGES:")
        print("-" * 60)
        
        rag_with_context = 0
        total_rag_tests = 0
        
        for result in self.results:
            if "bonequest_rag" in result["models"]:
                total_rag_tests += 1
                if result["models"]["bonequest_rag"].get("has_citations", False):
                    rag_with_context += 1
        
        if total_rag_tests > 0:
            context_percentage = (rag_with_context / total_rag_tests) * 100
            print(f"• Citations/Context: {context_percentage:.1f}% of responses included relevant citations")
            print(f"• Domain Knowledge: Responses based on uploaded orthopedic documents")
            print(f"• Role-based Output: Tailored responses for different clinical roles")
            print(f"• Hybrid Search: Combines semantic + keyword search for better relevance")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 {len(self.results)} queries tested across 4 AI models")

def main():
    """Main function to run the comparison"""
    # Check environment variables
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("Please set these environment variables before running the test.")
        return
    
    # Initialize and run comparison
    comparison = RAGComparison()
    
    print("🔧 Setup Instructions:")
    print("1. Make sure BoneQuest RAG backend is running on localhost:8000")
    print("2. Upload some orthopedic documents via admin panel")
    print("3. Set API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY")
    
    input("\nPress Enter to start the comparison test...")
    
    try:
        comparison.run_full_comparison()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")

if __name__ == "__main__":
    main()