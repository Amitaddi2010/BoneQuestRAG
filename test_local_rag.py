#!/usr/bin/env python3
"""
Local BoneQuest RAG Testing Framework
Tests only the BoneQuest system without external API dependencies
"""

import requests
import json
import time
from typing import Dict, List, Any

class LocalRAGTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_queries = [
            "What are the signs of a fracture in an X-ray?",
            "How do you treat a torn ACL?",
            "What is the difference between osteoarthritis and rheumatoid arthritis?",
            "Explain the anatomy of the knee joint",
            "What are the symptoms of a rotator cuff tear?",
            "How is a hip replacement performed?",
            "What causes lower back pain?",
            "Describe the healing process of bone fractures",
            "What are the risk factors for osteoporosis?",
            "How do you diagnose a meniscus tear?"
        ]
    
    def test_health_check(self) -> bool:
        """Test if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def test_chat_endpoint(self, query: str, role: str = "normal") -> Dict[str, Any]:
        """Test a single chat query"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "query": query,
                    "mode": role,
                    "model": "llama-3.1-8b-instant"
                }
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "sources": data.get("sources", []),
                    "response_time": end_time - start_time,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "response": "",
                    "sources": [],
                    "response_time": end_time - start_time,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "sources": [],
                "response_time": 0,
                "error": str(e)
            }
    
    def evaluate_response(self, query: str, response: str, sources: List[str]) -> Dict[str, float]:
        """Simple evaluation metrics for RAG responses"""
        metrics = {}
        
        # Response length (longer responses often more detailed)
        metrics["response_length"] = len(response.split())
        
        # Source utilization (how many sources were used)
        metrics["sources_used"] = len(sources)
        
        # Medical keyword coverage
        medical_keywords = [
            "bone", "joint", "muscle", "ligament", "tendon", "cartilage",
            "fracture", "injury", "pain", "treatment", "diagnosis", "symptom",
            "therapy", "surgery", "rehabilitation", "anatomy", "orthopedic"
        ]
        
        response_lower = response.lower()
        keyword_matches = sum(1 for keyword in medical_keywords if keyword in response_lower)
        metrics["medical_relevance"] = keyword_matches / len(medical_keywords)
        
        # Query relevance (simple keyword overlap)
        query_words = set(query.lower().split())
        response_words = set(response_lower.split())
        if query_words:
            metrics["query_relevance"] = len(query_words.intersection(response_words)) / len(query_words)
        else:
            metrics["query_relevance"] = 0
        
        return metrics
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive testing of BoneQuest RAG"""
        print("🧪 Starting BoneQuest RAG Local Testing...")
        
        # Health check
        if not self.test_health_check():
            return {"error": "Server not running at " + self.base_url}
        
        print("✅ Server is running")
        
        results = {
            "total_queries": len(self.test_queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0,
            "role_results": {},
            "detailed_results": []
        }
        
        roles = ["normal", "jr", "sr", "consultant"]
        
        for role in roles:
            print(f"\n🎭 Testing role: {role}")
            role_results = {
                "successful": 0,
                "failed": 0,
                "avg_response_time": 0,
                "avg_response_length": 0,
                "avg_sources_used": 0,
                "avg_medical_relevance": 0,
                "avg_query_relevance": 0
            }
            
            role_metrics = []
            
            for i, query in enumerate(self.test_queries):
                print(f"  Query {i+1}/{len(self.test_queries)}: {query[:50]}...")
                
                result = self.test_chat_endpoint(query, role)
                
                if result["success"]:
                    role_results["successful"] += 1
                    results["successful_queries"] += 1
                    
                    # Evaluate response
                    metrics = self.evaluate_response(
                        query, 
                        result["response"], 
                        result["sources"]
                    )
                    
                    role_metrics.append({
                        "response_time": result["response_time"],
                        **metrics
                    })
                    
                    results["detailed_results"].append({
                        "query": query,
                        "role": role,
                        "success": True,
                        "response_length": len(result["response"]),
                        "sources_count": len(result["sources"]),
                        "response_time": result["response_time"],
                        **metrics
                    })
                    
                else:
                    role_results["failed"] += 1
                    results["failed_queries"] += 1
                    print(f"    ❌ Failed: {result['error']}")
                    
                    results["detailed_results"].append({
                        "query": query,
                        "role": role,
                        "success": False,
                        "error": result["error"]
                    })
            
            # Calculate averages for this role
            if role_metrics:
                role_results["avg_response_time"] = sum(m["response_time"] for m in role_metrics) / len(role_metrics)
                role_results["avg_response_length"] = sum(m["response_length"] for m in role_metrics) / len(role_metrics)
                role_results["avg_sources_used"] = sum(m["sources_used"] for m in role_metrics) / len(role_metrics)
                role_results["avg_medical_relevance"] = sum(m["medical_relevance"] for m in role_metrics) / len(role_metrics)
                role_results["avg_query_relevance"] = sum(m["query_relevance"] for m in role_metrics) / len(role_metrics)
            
            results["role_results"][role] = role_results
        
        # Calculate overall averages
        successful_results = [r for r in results["detailed_results"] if r["success"]]
        if successful_results:
            results["average_response_time"] = sum(r["response_time"] for r in successful_results) / len(successful_results)
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("🎯 BONEQUEST RAG TEST RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total Queries: {results['total_queries']}")
        print(f"  Successful: {results['successful_queries']} ✅")
        print(f"  Failed: {results['failed_queries']} ❌")
        print(f"  Success Rate: {(results['successful_queries']/results['total_queries']*100):.1f}%")
        print(f"  Average Response Time: {results['average_response_time']:.2f}s")
        
        print(f"\n🎭 ROLE-BASED PERFORMANCE:")
        role_names = {
            "normal": "👤 Patient",
            "jr": "🎓 Junior Resident", 
            "sr": "👨⚕️ Senior Resident",
            "consultant": "🩺 Consultant"
        }
        
        for role, data in results["role_results"].items():
            print(f"\n  {role_names.get(role, role)}:")
            print(f"    Success Rate: {(data['successful']/(data['successful']+data['failed'])*100):.1f}%")
            print(f"    Avg Response Time: {data['avg_response_time']:.2f}s")
            print(f"    Avg Response Length: {data['avg_response_length']:.0f} words")
            print(f"    Avg Sources Used: {data['avg_sources_used']:.1f}")
            print(f"    Medical Relevance: {data['avg_medical_relevance']*100:.1f}%")
            print(f"    Query Relevance: {data['avg_query_relevance']*100:.1f}%")
        
        print(f"\n🔍 TOP PERFORMING QUERIES:")
        successful = [r for r in results["detailed_results"] if r["success"]]
        if successful:
            top_queries = sorted(successful, key=lambda x: x["medical_relevance"], reverse=True)[:3]
            for i, query in enumerate(top_queries, 1):
                print(f"  {i}. {query['query'][:50]}...")
                print(f"     Medical Relevance: {query['medical_relevance']*100:.1f}%")
        
        print("\n" + "="*60)

def main():
    """Main testing function"""
    tester = LocalRAGTester()
    
    print("🚀 BoneQuest RAG Local Testing Framework")
    print("This will test your local BoneQuest system only")
    print("Make sure your server is running on http://localhost:8000")
    
    input("\nPress Enter to start testing...")
    
    results = tester.run_comprehensive_test()
    tester.print_results(results)
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: test_results.json")

if __name__ == "__main__":
    main()