@echo off
echo Setting up BoneQuest RAG Comparison Test Environment
echo.

echo Installing test dependencies...
pip install -r requirements_test.txt

echo.
echo Please set your API keys as environment variables:
echo.
echo set OPENAI_API_KEY=your_openai_api_key_here
echo set ANTHROPIC_API_KEY=your_anthropic_api_key_here  
echo set GROQ_API_KEY=your_groq_api_key_here
echo.
echo Then run: python test_rag_comparison.py
echo.
pause