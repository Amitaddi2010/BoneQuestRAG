import subprocess
import sys

# Uninstall and reinstall groq with correct version
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'groq', '-y'])
subprocess.run([sys.executable, '-m', 'pip', 'install', 'groq==0.9.0'])

print("Groq fixed! Now run: python backend/main.py")