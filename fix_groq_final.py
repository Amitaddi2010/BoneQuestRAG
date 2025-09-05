import subprocess
import sys

# Try different groq versions
versions = ['0.4.2', '0.3.0', '0.2.0']

for version in versions:
    print(f"Trying groq version {version}...")
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', 'groq', '-y'], capture_output=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', f'groq=={version}'], capture_output=True)
    
    # Test this version
    try:
        from groq import Groq
        client = Groq(api_key="test")
        print(f"Version {version} works!")
        break
    except Exception as e:
        print(f"Version {version} failed: {e}")
        continue

print("Done. Test with: python test_groq.py")