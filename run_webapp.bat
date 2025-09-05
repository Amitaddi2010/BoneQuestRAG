@echo off
echo Starting BoneQuest RAG WebApp...
echo.

echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
echo.

echo Starting FastAPI backend...
start "BoneQuest Backend" cmd /k "python main.py"
echo Backend started on http://localhost:8000
echo.

echo Starting frontend...
cd ..\frontend
start "BoneQuest Frontend" index.html
echo Frontend opened in browser
echo.

echo BoneQuest WebApp is now running!
echo Backend: http://localhost:8000
echo Frontend: Open index.html in browser
pause