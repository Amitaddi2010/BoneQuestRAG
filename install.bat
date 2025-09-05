@echo off
echo Installing BoneQuest dependencies...
pip install --upgrade pip
pip install streamlit groq pypdf2 python-dotenv numpy scipy scikit-learn
echo.
echo Installation complete!
echo Run BoneQuest with: streamlit run app.py
pause