import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
print(f"API Key: {api_key}")

try:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="llama3-8b-8192",
        max_tokens=50
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)