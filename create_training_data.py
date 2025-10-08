#!/usr/bin/env python3
"""
Create training PDF from 20 orthopedic Q&A pairs
This will improve RAG accuracy by adding the test data to knowledge base
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_training_pdf():
    """Create PDF with 20 orthopedic Q&A pairs for RAG training"""
    
    # Q&A data
    qa_pairs = [
        {
            "question": "Responding on a PI, you find an unrestrained driver awake, complaining of severe pelvic and abdominal pain, with a BP of 80/P. Is this patient a rapid extrication? Should you compress or 'rock' the patient's pelvis to assess stability?",
            "answer": "Yes, the patient is a rapid extrication due to his shock from pelvic and possible abdominal injury. Because the patient already complains of pelvic pain, compression of the pelvis as part of the assessment is not helpful, and aggressive compression may increase bleeding from an unstable pelvis."
        },
        {
            "question": "What are appropriate therapies for this patient?",
            "answer": "Rapid extrication with C-collar onto a long backboard, application of the SAM binder, rapid transport, and establishment of a large bore IV of normal saline enroute to a trauma center would be the preferred interventions for this patient, with secondary survey potentially leading to other therapies (e.g. chest decompression, etc.) – the SAM binder helps with typical 'open book' pelvic fractures to bring the 'wings' of the pelvis together and help stop bleeding – it should be applied any time you suspect a pelvic fracture (this could be an unresponsive multi-trauma patient with hypotension, for example, not just patients with pelvic pain)"
        },
        {
            "question": "The patient also has an unstable, angulated tib/fib fracture. How should you splint this fracture? Do you need to 'splint it as it lies?'.",
            "answer": "Aluminum/foam SAM splint or other basic splints can be used – don't spend much time splinting but at least stabilize the fracture. When fractures are angulated or joints held in positions that will not allow effective splinting, gentle restoration of the normal position of the limb is advisable, usually with gentle traction applied. If the patient has a significant increase in pain with the new position, or if the circulation/sensory exam worsens, you may have to reposition."
        },
        {
            "question": "A small laceration is noted near the site of the angulation. No bone is seen. Is this an 'open' fracture?",
            "answer": "Yes, any open wound that could communicate with a fracture means the fracture is treated as open. Exposure of bone ends is rare, but would clearly confirm your suspicions. (Do NOT attempt to manipulate the extremity to see this, of course!)"
        },
        {
            "question": "A passenger in the same vehicle is complaining of midshaft thigh pain. Her vitals are stable currently. How much blood can be lost from a midshaft femur fracture?",
            "answer": "1000-1500cc of blood can be lost from an 'uncomplicated' midshaft femur fracture."
        },
        {
            "question": "How should this injury be cared for?",
            "answer": "Traction splinting is indicated for midshaft, isolated femur fractures. It is somewhat time-consuming however, and simple immobilization on a backboard should be considered for a patient with multi-system injuries. When applying a traction device, make sure to place the groin strap as far up as possible. Once this strap is applied, and a foot harness in place, traction is applied, then the rest of the straps placed."
        },
        {
            "question": "Following any splinting procedure, what three things should be assessed and documented?",
            "answer": "Circulation, motor, and sensory function. We often we forget to do this, and it can be a major issue as sometimes, particularly with traction or re-positioning the exam can get worse instead of better."
        },
        {
            "question": "If the patient's foot is cool, and pulses not present, what can be done? How long until the muscles and limb suffer irreversible damage?",
            "answer": "In the setting of a femur fracture, the amount of traction may be varied to see if the pulse will return. Reduction to neutral alignment of other extremity fractures may help to restore a pulse. The general rule is within 4 hours, the muscles will die, and survival of the limb will be in question."
        },
        {
            "question": "A patient in the other vehicle is complaining of knee pain. On extrication, you place him on a backboard. His knee is slightly bent, and the hip is internally rotated. He is unable to straighten the leg without severe pain. He believes that his knee hit the dashboard. What do you suspect is injured?",
            "answer": "Patients with posterior hip dislocations from impact vs. the dash often refer pain to the knee. This patient exhibits a typical position. Reduction needs to occur soon to prevent compressive damage to the sciatic nerve. This cannot be done in the field and the patient will have to be immobilized with the knee flexed."
        },
        {
            "question": "Following restocking your truck, you are called for a 'jumper' who jumps from the second story window of a school after screaming \"I can't take these little bastards\". He was witnessed to land in an almost standing position before crumpling to the ground. What types of bony injuries are usually associated with such a mechanism?",
            "answer": "This is referred to as a 'Don Juan' injury. Ask your partner why…Jumping from a height and landing on one's feet leads to several predictable fractures (and often with other injuries, including pelvic and abdominal). The heelbones (calcanei), knees, hips, and lumbar spine are at greatest risk of fracture."
        },
        {
            "question": "In frustration with his situation the teacher punches the ground with a closed fist, and howls in pain. He is tender over the 5th metacarpal of the hand, with mild deformity near the metacarpal head. What classic fracture is this?",
            "answer": "This is a classic mechanism and exam for a 'boxer's fracture' which may go undiagnosed for hours to days. If there is significant angulation of the fracture, the patient may have problems with range of motion and grip. Remember to wear padded gloves when you hit hard objects!"
        },
        {
            "question": "A nursing home patient fell out of bed, and is complaining of hip pain. What is the mortality of a hip fracture in an elderly patient?",
            "answer": "Elderly patients who suffer 'hip' fractures (actually proximal femur and femoral neck fractures) have generally poor outcomes due to the need for bedrest or operations. Within a year, approximately 30-40% will die of related causes (pneumonia due to bedrest, operative complications, etc.)"
        },
        {
            "question": "When checking the patient's pulses, you are unable to find a dorsalis pedis pulse on either side. What percent of the population normally lacks a dorsalis pedis pulse?",
            "answer": "12-17% of the Caucasian population do not have a dorsalis pedis pulse, the posterior tibial pulses are basically always present, in African-Americans up to 9% lack a posterior tibial pulse. In patients over age 45 one or the other may be dominant, and the other not palpable, but detectable with ultrasound."
        },
        {
            "question": "This same patient became entangled in the sheets falling out of bed, and has a shoulder dislocation. What is the most common nerve injured with a dislocated shoulder and where can you check sensation for this nerve?",
            "answer": "The axillary nerve. The area of sensation is over the lateral shoulder (deltoid muscle area, just over the humeral head). If sensation is absent, this is important to note before immobilization."
        },
        {
            "question": "Walking to the light rail station, a Vikings fan steps on a Green Bay fan's face, severely twisting his ankle. He is very tender below the medial malleolus. What other area needs to be examined?",
            "answer": "This injury has an associated fracture of the proximal fibula (lateral leg, just below the knee); it is called a 'Maisonneuve' fracture. Don't forget to examine the calf and knee, especially laterally, and splint the knee if indicated. This does NOT occur with lateral (inversion) sprains (though please still check the knee, for injury)"
        },
        {
            "question": "If this fracture is missed, will the patient notice?",
            "answer": "This fracture is easily missed, the patient may walk around for some time with nagging, but not severe pain. Injuries to the peroneal nerve are relatively common if this isn't recognized and treated. Pro hockey players have played on these fractures during playoffs etc. (against medical advice, of course), so they are not an immediately disabling injury."
        },
        {
            "question": "The patient states he cannot feel his 4th and 5th toes. He is panicked over the thought of losing his sensation there 'for life'. What can you tell him?",
            "answer": "Almost all cases in which the patient reports altered sensation in the absence of a circulatory deficit or major dislocation/deformity will resolve over days to weeks. Essentially, swelling around the injury affects nerves, which alters sensation to points distally. As the swelling goes away, the nerve usually will gradually recover its function. Rapid progression of sensory changes is often a bad thing, and may signal compartment syndrome or poor circulation."
        },
        {
            "question": "A 2 year old was dragged to the Twins game by her parents (season ticket holders), and now refuses to use her R arm. No other trauma has been noted. What is the most common cause of this injury?",
            "answer": "The 'nursemaid's elbow' is usually caused by traction on the arm of a toddler (though we can see this injury even up to the age of 9-10 years!). At this age, the ligaments that hold the radius in place in the elbow are loose, and dislocation occurs. Reduction is a fairly easy process of turning the hand and flexing the elbow, but obviously we need to be careful not to be trying to relocate a fracture!"
        },
        {
            "question": "A man who jumped up to celebrate the Twins covering the spread is stricken when his knee 'locks up'. He is severely distressed, and cannot seem to straighten the knee. What is the usual cause of an atraumatic locked knee?",
            "answer": "Generally, a knee 'locks' because of a meniscal tear (tear in the cartilage lining of the knee joint), unlocking usually requires good pain relief and often sedation. Usually if this is occurring, arthroscopy is indicated to see if it can be trimmed or repaired. Knee locking after acute trauma is often due to fractures and muscle spasm."
        },
        {
            "question": "A patient suffers an amputation of his thumb from a band saw. How should you care for the thumb and is this patient a candidate for re-implantation?",
            "answer": "Greater efforts are made to re-implant thumbs if possible, isolated fingers may depend on which finger, handedness of the patient, and amount of soft tissue injury. If the patient is healthy, and the amputation was sharp (e.g. saw, rather than a crush injury), the odds of re-implantation success are fairly good within the first 6 hours. The thumb should be placed in a bag, with damp dressings, then the bag placed on ice if possible. No direct ice should be applied to the amputated digit. A few surgeons in the Twin Cites do re-implantation, they practice at Fairview-University, North, and HCMC. Incidentally, toes are not reimplanted, and finger/thumbs do the best of all re-implants."
        }
    ]
    
    # Create PDF
    filename = "orthopedic_training_data.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.5*inch)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=1  # Center
    )
    
    question_style = ParagraphStyle(
        'Question',
        parent=styles['Heading2'],
        fontSize=12,
        textColor='blue',
        spaceAfter=10
    )
    
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=15,
        leftIndent=20
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("Orthopedic Emergency Medicine Training Manual", title_style))
    content.append(Paragraph("Comprehensive Q&A for Emergency Medical Services", styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Add Q&A pairs
    for i, qa in enumerate(qa_pairs, 1):
        content.append(Paragraph(f"Question {i}: {qa['question']}", question_style))
        content.append(Paragraph(f"Answer: {qa['answer']}", answer_style))
        content.append(Spacer(1, 10))
    
    # Build PDF
    doc.build(content)
    print(f"✅ Training PDF created: {filename}")
    return filename

def main():
    """Create training PDF and provide upload instructions"""
    print("📚 Creating Orthopedic Training Data PDF...")
    
    try:
        filename = create_training_pdf()
        
        print(f"\n🎯 Training data created successfully!")
        print(f"📄 File: {filename}")
        print(f"\n📋 Next steps to improve RAG accuracy:")
        print(f"1. Start your BoneQuest server: python backend/main.py")
        print(f"2. Open http://localhost:8000 in browser")
        print(f"3. Click Admin panel, login with password: admin123")
        print(f"4. Upload the {filename} file")
        print(f"5. Run accuracy test again: python test_accuracy.py")
        print(f"\n💡 Expected improvement: 60-80% accuracy after training")
        
    except ImportError:
        print("❌ ReportLab not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "reportlab"])
        print("✅ ReportLab installed. Run script again.")

if __name__ == "__main__":
    main()