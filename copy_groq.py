import shutil
import os
import sys

# Copy working groq from streamlit environment
streamlit_groq = None
for path in sys.path:
    groq_path = os.path.join(path, 'groq')
    if os.path.exists(groq_path):
        print(f"Found groq at: {groq_path}")
        break

# Install exact same version as streamlit
import subprocess
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'groq', '-y'])
subprocess.run([sys.executable, '-m', 'pip', 'install', 'groq>=0.4.0', '--force-reinstall'])

print("Groq reinstalled with streamlit-compatible version")