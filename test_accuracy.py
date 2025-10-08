#!/usr/bin/env python3
"""
BoneQuest RAG Accuracy Testing Framework
Tests accuracy against 20 known orthopedic Q&A pairs
"""

import requests
import json
import time
from typing import Dict, List, Tuple
from difflib import SequenceMatcher

class AccuracyTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
        # 20 Test Questions and Expected Answers
        self.test_data = [
            {
                "question": "Responding on a PI, you find an unrestrained driver awake, complaining of severe pelvic and abdominal pain, with a BP of 80/P. Is this patient a rapid extrication? Should you compress or 'rock' the patient's pelvis to assess stability?",
                "expected": "Yes, the patient is a rapid extrication due to his shock from pelvic and possible abdominal injury. Because the patient already complains of pelvic pain, compression of the pelvis as part of the assessment is not helpful, and aggressive compression may increase bleeding from an unstable pelvis."
            },
            {
                "question": "What are appropriate therapies for this patient?",
                "expected": "Rapid extrication with C-collar onto a long backboard, application of the SAM binder, rapid transport, and establishment of a large bore IV of normal saline enroute to a trauma center would be the preferred interventions for this patient, with secondary survey potentially leading to other therapies (e.g. chest decompression, etc.) – the SAM binder helps with typical 'open book' pelvic fractures to bring the 'wings' of the pelvis together and help stop bleeding – it should be applied any time you suspect a pelvic fracture (this could be an unresponsive multi-trauma patient with hypotension, for example, not just patients with pelvic pain)"
            },
            {
                "question": "The patient also has an unstable, angulated tib/fib fracture. How should you splint this fracture? Do you need to 'splint it as it lies?'.",
                "expected": "Aluminum/foam SAM splint or other basic splints can be used – don't spend much time splinting but at least stabilize the fracture. When fractures are angulated or joints held in positions that will not allow effective splinting, gentle restoration of the normal position of the limb is advisable, usually with gentle traction applied. If the patient has a significant increase in pain with the new position, or if the circulation/sensory exam worsens, you may have to reposition."
            },
            {
                "question": "A small laceration is noted near the site of the angulation. No bone is seen. Is this an 'open' fracture?",
                "expected": "Yes, any open wound that could communicate with a fracture means the fracture is treated as open. Exposure of bone ends is rare, but would clearly confirm your suspicions. (Do NOT attempt to manipulate the extremity to see this, of course!)"
            },
            {
                "question": "A passenger in the same vehicle is complaining of midshaft thigh pain. Her vitals are stable currently. How much blood can be lost from a midshaft femur fracture?",
                "expected": "1000-1500cc of blood can be lost from an 'uncomplicated' midshaft femur fracture."
            },
            {
                "question": "How should this injury be cared for?",
                "expected": "Traction splinting is indicated for midshaft, isolated femur fractures. It is somewhat time-consuming however, and simple immobilization on a backboard should be considered for a patient with multi-system injuries. When applying a traction device, make sure to place the groin strap as far up as possible. Once this strap is applied, and a foot harness in place, traction is applied, then the rest of the straps placed."
            },
            {
                "question": "Following any splinting procedure, what three things should be assessed and documented?",
                "expected": "Circulation, motor, and sensory function. We often we forget to do this, and it can be a major issue as sometimes, particularly with traction or re-positioning the exam can get worse instead of better."
            },
            {
                "question": "If the patient's foot is cool, and pulses not present, what can be done? How long until the muscles and limb suffer irreversible damage?",
                "expected": "In the setting of a femur fracture, the amount of traction may be varied to see if the pulse will return. Reduction to neutral alignment of other extremity fractures may help to restore a pulse. The general rule is within 4 hours, the muscles will die, and survival of the limb will be in question."
            },
            {
                "question": "A patient in the other vehicle is complaining of knee pain. On extrication, you place him on a backboard. His knee is slightly bent, and the hip is internally rotated. He is unable to straighten the leg without severe pain. He believes that his knee hit the dashboard. What do you suspect is injured?",
                "expected": "Patients with posterior hip dislocations from impact vs. the dash often refer pain to the knee. This patient exhibits a typical position. Reduction needs to occur soon to prevent compressive damage to the sciatic nerve. This cannot be done in the field and the patient will have to be immobilized with the knee flexed."
            },
            {
                "question": "Following restocking your truck, you are called for a 'jumper' who jumps from the second story window of a school after screaming \"I can't take these little bastards\". He was witnessed to land in an almost standing position before crumpling to the ground. What types of bony injuries are usually associated with such a mechanism?",
                "expected": "This is referred to as a 'Don Juan' injury. Ask your partner why…Jumping from a height and landing on one's feet leads to several predictable fractures (and often with other injuries, including pelvic and abdominal). The heelbones (calcanei), knees, hips, and lumbar spine are at greatest risk of fracture."
            },
            {
                "question": "In frustration with his situation the teacher punches the ground with a closed fist, and howls in pain. He is tender over the 5th metacarpal of the hand, with mild deformity near the metacarpal head. What classic fracture is this?",
                "expected": "This is a classic mechanism and exam for a 'boxer's fracture' which may go undiagnosed for hours to days. If there is significant angulation of the fracture, the patient may have problems with range of motion and grip. Remember to wear padded gloves when you hit hard objects!"
            },
            {
                "question": "A nursing home patient fell out of bed, and is complaining of hip pain. What is the mortality of a hip fracture in an elderly patient?",
                "expected": "Elderly patients who suffer 'hip' fractures (actually proximal femur and femoral neck fractures) have generally poor outcomes due to the need for bedrest or operations. Within a year, approximately 30-40% will die of related causes (pneumonia due to bedrest, operative complications, etc.)"
            },
            {
                "question": "When checking the patient's pulses, you are unable to find a dorsalis pedis pulse on either side. What percent of the population normally lacks a dorsalis pedis pulse?",
                "expected": "12-17% of the Caucasian population do not have a dorsalis pedis pulse, the posterior tibial pulses are basically always present, in African-Americans up to 9% lack a posterior tibial pulse. In patients over age 45 one or the other may be dominant, and the other not palpable, but detectable with ultrasound."
            },
            {
                "question": "This same patient became entangled in the sheets falling out of bed, and has a shoulder dislocation. What is the most common nerve injured with a dislocated shoulder and where can you check sensation for this nerve?",
                "expected": "The axillary nerve. The area of sensation is over the lateral shoulder (deltoid muscle area, just over the humeral head). If sensation is absent, this is important to note before immobilization."
            },
            {
                "question": "Walking to the light rail station, a Vikings fan steps on a Green Bay fan's face, severely twisting his ankle. He is very tender below the medial malleolus. What other area needs to be examined?",
                "expected": "This injury has an associated fracture of the proximal fibula (lateral leg, just below the knee); it is called a 'Maisonneuve' fracture. Don't forget to examine the calf and knee, especially laterally, and splint the knee if indicated. This does NOT occur with lateral (inversion) sprains (though please still check the knee, for injury)"
            },
            {
                "question": "If this fracture is missed, will the patient notice?",
                "expected": "This fracture is easily missed, the patient may walk around for some time with nagging, but not severe pain. Injuries to the peroneal nerve are relatively common if this isn't recognized and treated. Pro hockey players have played on these fractures during playoffs etc. (against medical advice, of course), so they are not an immediately disabling injury."
            },
            {
                "question": "The patient states he cannot feel his 4th and 5th toes. He is panicked over the thought of losing his sensation there 'for life'. What can you tell him?",
                "expected": "Almost all cases in which the patient reports altered sensation in the absence of a circulatory deficit or major dislocation/deformity will resolve over days to weeks. Essentially, swelling around the injury affects nerves, which alters sensation to points distally. As the swelling goes away, the nerve usually will gradually recover its function. Rapid progression of sensory changes is often a bad thing, and may signal compartment syndrome or poor circulation."
            },
            {
                "question": "A 2 year old was dragged to the Twins game by her parents (season ticket holders), and now refuses to use her R arm. No other trauma has been noted. What is the most common cause of this injury?",
                "expected": "The 'nursemaid's elbow' is usually caused by traction on the arm of a toddler (though we can see this injury even up to the age of 9-10 years!). At this age, the ligaments that hold the radius in place in the elbow are loose, and dislocation occurs. Reduction is a fairly easy process of turning the hand and flexing the elbow, but obviously we need to be careful not to be trying to relocate a fracture!"
            },
            {
                "question": "A man who jumped up to celebrate the Twins covering the spread is stricken when his knee 'locks up'. He is severely distressed, and cannot seem to straighten the knee. What is the usual cause of an atraumatic locked knee?",
                "expected": "Generally, a knee 'locks' because of a meniscal tear (tear in the cartilage lining of the knee joint), unlocking usually requires good pain relief and often sedation. Usually if this is occurring, arthroscopy is indicated to see if it can be trimmed or repaired. Knee locking after acute trauma is often due to fractures and muscle spasm."
            },
            {
                "question": "A patient suffers an amputation of his thumb from a band saw. How should you care for the thumb and is this patient a candidate for re-implantation?",
                "expected": "Greater efforts are made to re-implant thumbs if possible, isolated fingers may depend on which finger, handedness of the patient, and amount of soft tissue injury. If the patient is healthy, and the amputation was sharp (e.g. saw, rather than a crush injury), the odds of re-implantation success are fairly good within the first 6 hours. The thumb should be placed in a bag, with damp dressings, then the bag placed on ice if possible. No direct ice should be applied to the amputated digit. A few surgeons in the Twin Cites do re-implantation, they practice at Fairview-University, North, and HCMC. Incidentally, toes are not reimplanted, and finger/thumbs do the best of all re-implants."
            }
        ]
    
    def test_health(self) -> bool:
        """Test server health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def query_rag(self, question: str, mode: str = "sr") -> Dict:
        """Query the RAG system"""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "query": question,
                    "mode": mode,
                    "model": "llama-3.1-8b-instant"
                }
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "context": data.get("context", []),
                    "response_time": end_time - start_time
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response_time": end_time - start_time
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key medical terms from text"""
        medical_terms = [
            "fracture", "dislocation", "nerve", "artery", "vein", "bone", "joint",
            "ligament", "tendon", "muscle", "blood", "pulse", "circulation",
            "traction", "splint", "immobilization", "reduction", "amputation",
            "extrication", "trauma", "injury", "pain", "swelling", "deformity"
        ]
        
        text_lower = text.lower()
        found_terms = []
        for term in medical_terms:
            if term in text_lower:
                found_terms.append(term)
        return found_terms
    
    def evaluate_answer(self, question: str, rag_response: str, expected: str) -> Dict:
        """Evaluate RAG response against expected answer"""
        
        # Text similarity
        similarity = self.calculate_similarity(rag_response, expected)
        
        # Key term coverage
        expected_terms = set(self.extract_key_terms(expected))
        rag_terms = set(self.extract_key_terms(rag_response))
        
        if expected_terms:
            term_coverage = len(expected_terms.intersection(rag_terms)) / len(expected_terms)
        else:
            term_coverage = 0
        
        # Response completeness (length comparison)
        length_ratio = min(len(rag_response) / len(expected), 1.0) if expected else 0
        
        # Overall accuracy score (weighted average)
        accuracy_score = (similarity * 0.5) + (term_coverage * 0.3) + (length_ratio * 0.2)
        
        return {
            "similarity": similarity,
            "term_coverage": term_coverage,
            "length_ratio": length_ratio,
            "accuracy_score": accuracy_score,
            "expected_terms": list(expected_terms),
            "rag_terms": list(rag_terms),
            "missing_terms": list(expected_terms - rag_terms),
            "extra_terms": list(rag_terms - expected_terms)
        }
    
    def run_accuracy_test(self) -> Dict:
        """Run comprehensive accuracy testing"""
        print("🎯 Starting BoneQuest RAG Accuracy Testing...")
        
        if not self.test_health():
            return {"error": "Server not running at " + self.base_url}
        
        print("✅ Server is running")
        
        results = {
            "total_questions": len(self.test_data),
            "successful_queries": 0,
            "failed_queries": 0,
            "average_accuracy": 0,
            "average_similarity": 0,
            "average_term_coverage": 0,
            "average_response_time": 0,
            "detailed_results": [],
            "top_performers": [],
            "needs_improvement": []
        }
        
        total_accuracy = 0
        total_similarity = 0
        total_term_coverage = 0
        total_response_time = 0
        
        for i, test_case in enumerate(self.test_data):
            print(f"\n📝 Question {i+1}/{len(self.test_data)}")
            print(f"Q: {test_case['question'][:80]}...")
            
            # Query RAG system
            rag_result = self.query_rag(test_case["question"])
            
            if rag_result["success"]:
                results["successful_queries"] += 1
                
                # Evaluate accuracy
                evaluation = self.evaluate_answer(
                    test_case["question"],
                    rag_result["response"],
                    test_case["expected"]
                )
                
                total_accuracy += evaluation["accuracy_score"]
                total_similarity += evaluation["similarity"]
                total_term_coverage += evaluation["term_coverage"]
                total_response_time += rag_result["response_time"]
                
                # Store detailed result
                detailed_result = {
                    "question_id": i + 1,
                    "question": test_case["question"],
                    "expected_answer": test_case["expected"],
                    "rag_response": rag_result["response"],
                    "context_sources": len(rag_result.get("context", [])),
                    "response_time": rag_result["response_time"],
                    **evaluation
                }
                
                results["detailed_results"].append(detailed_result)
                
                # Categorize performance
                if evaluation["accuracy_score"] >= 0.7:
                    results["top_performers"].append(detailed_result)
                elif evaluation["accuracy_score"] < 0.4:
                    results["needs_improvement"].append(detailed_result)
                
                print(f"✅ Accuracy: {evaluation['accuracy_score']:.2f}")
                print(f"   Similarity: {evaluation['similarity']:.2f}")
                print(f"   Term Coverage: {evaluation['term_coverage']:.2f}")
                
            else:
                results["failed_queries"] += 1
                print(f"❌ Failed: {rag_result['error']}")
                
                results["detailed_results"].append({
                    "question_id": i + 1,
                    "question": test_case["question"],
                    "error": rag_result["error"],
                    "success": False
                })
        
        # Calculate averages
        if results["successful_queries"] > 0:
            results["average_accuracy"] = total_accuracy / results["successful_queries"]
            results["average_similarity"] = total_similarity / results["successful_queries"]
            results["average_term_coverage"] = total_term_coverage / results["successful_queries"]
            results["average_response_time"] = total_response_time / results["successful_queries"]
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted accuracy test results"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*70)
        print("🎯 BONEQUEST RAG ACCURACY TEST RESULTS")
        print("="*70)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Questions: {results['total_questions']}")
        print(f"  Successful: {results['successful_queries']} ✅")
        print(f"  Failed: {results['failed_queries']} ❌")
        print(f"  Success Rate: {(results['successful_queries']/results['total_questions']*100):.1f}%")
        
        if results['successful_queries'] > 0:
            print(f"\n🎯 ACCURACY METRICS:")
            print(f"  Overall Accuracy: {results['average_accuracy']*100:.1f}%")
            print(f"  Text Similarity: {results['average_similarity']*100:.1f}%")
            print(f"  Medical Term Coverage: {results['average_term_coverage']*100:.1f}%")
            print(f"  Average Response Time: {results['average_response_time']:.2f}s")
            
            # Performance categories
            accuracy_grade = "A" if results['average_accuracy'] >= 0.8 else \
                           "B" if results['average_accuracy'] >= 0.6 else \
                           "C" if results['average_accuracy'] >= 0.4 else "D"
            
            print(f"  Performance Grade: {accuracy_grade}")
        
        print(f"\n🏆 TOP PERFORMING QUESTIONS ({len(results['top_performers'])}):")
        for result in results['top_performers'][:5]:
            print(f"  Q{result['question_id']}: {result['accuracy_score']*100:.1f}% - {result['question'][:60]}...")
        
        print(f"\n⚠️  NEEDS IMPROVEMENT ({len(results['needs_improvement'])}):")
        for result in results['needs_improvement'][:5]:
            print(f"  Q{result['question_id']}: {result['accuracy_score']*100:.1f}% - {result['question'][:60]}...")
            if result.get('missing_terms'):
                print(f"    Missing terms: {', '.join(result['missing_terms'][:5])}")
        
        print("\n" + "="*70)

def main():
    """Main testing function"""
    tester = AccuracyTester()
    
    print("🎯 BoneQuest RAG Accuracy Testing Framework")
    print("Testing against 20 known orthopedic Q&A pairs")
    print("Make sure your server is running on http://localhost:8000")
    
    input("\nPress Enter to start accuracy testing...")
    
    results = tester.run_accuracy_test()
    tester.print_results(results)
    
    # Save results
    with open("accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: accuracy_results.json")

if __name__ == "__main__":
    main()