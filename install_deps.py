import subprocess
import sys

deps = [
    'fastapi==0.104.1',
    'uvicorn==0.24.0', 
    'python-multipart==0.0.6',
    'groq==0.4.1',
    'PyPDF2==3.0.1',
    'scikit-learn==1.3.2',
    'numpy==1.24.3',
    'python-dotenv==1.0.0'
]

for dep in deps:
    subprocess.run([sys.executable, '-m', 'pip', 'install', dep])

print("All dependencies installed!")
print("Now run: python backend/main.py")