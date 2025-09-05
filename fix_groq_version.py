import subprocess
import sys

# Fix groq version compatibility
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'groq', '-y'])
subprocess.run([sys.executable, '-m', 'pip', 'install', 'groq==0.8.0'])

print("Groq version fixed! Now restart backend.")