import sys
import io
from pathlib import Path
from typing import Dict

sys.path.append(".")

import pdfplumber
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

from src.pipeline.rag_hybrid import prepare_contract, answer_question

app = FastAPI(title="Legal Contract Assistant")

TEMPLATE_DIR = Path(__file__).parent / "templates"
user_sessions: Dict[str, dict] = {}

@app.get("/", response_class=HTMLResponse)
async def home():
    return (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    content = await file.read()

    text = ""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from this PDF.")

    prepared = prepare_contract(text)
    user_sessions[session_id] = prepared

    return {
        "message": f"{file.filename} uploaded and analyzed ({len(prepared['chunks'])} chunks).",
        "chunks": len(prepared["chunks"])
    }

@app.post("/ask")
async def ask(data: dict):
    session_id = data.get("session_id")
    question = data.get("question")

    if not session_id or not question:
        raise HTTPException(status_code=400, detail="Missing session_id or question.")

    prepared = user_sessions.get(session_id)
    if prepared is None:
        raise HTTPException(status_code=400, detail="No contract uploaded for this session.")

    try:
        answer = answer_question(prepared, question, k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    return {"answer": answer}