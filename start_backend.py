import subprocess
import sys
import os

# Change to backend directory
os.chdir('backend')

# Install requirements
print("Installing backend dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

# Start the server
print("Starting BoneQuest backend server...")
subprocess.run([sys.executable, 'main.py'])