@echo off
REM ==========================================================
REM  RAG Q&A Assistant - one-click launcher (web UI)
REM  Double-click this file. Browser opens http://localhost:8501
REM  To stop: switch back to this window and press Ctrl+C
REM ==========================================================
cd /d "%~dp0"
set HF_ENDPOINT=https://hf-mirror.com
echo Starting RAG Q&A Assistant...
echo Open http://localhost:8501 in your browser when ready.
echo Press Ctrl+C to stop.
echo ----------------------------------------------------------
"D:\ana\envs\rag_project\python.exe" -m streamlit run app.py
pause
