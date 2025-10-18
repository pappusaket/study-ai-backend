from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import hashlib
import io
from pypdf import PdfReader
import os
import sqlite3
from datetime import datetime
import requests
import json

app = FastAPI(title="Study AI - Pure Python")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = "study-ai-index-jjuj0dk.svc.aped-4627-b74a.pinecone.io"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Database setup
DB_PATH = "study_ai.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            description TEXT,
            upload_date TEXT NOT NULL,
            status TEXT NOT NULL,
            file_size INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"

class PDFInfo(BaseModel):
    id: int
    filename: str
    description: str
    upload_date: str
    status: str
    file_size: int

# Utility Functions
def extract_text_from_pdf(pdf_content):
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF reading error: {str(e)}")

def add_pdf_to_db(filename, file_hash, user_id, description, file_size):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO pdf_files 
            (filename, file_hash, user_id, description, upload_date, status, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, file_hash, user_id, description, datetime.now().isoformat(), 'uploaded', file_size))
        conn.commit()
        pdf_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        cursor.execute('SELECT id FROM pdf_files WHERE file_hash = ?', (file_hash,))
        result = cursor.fetchone()
        pdf_id = result[0] if result else None
    finally:
        conn.close()
    return pdf_id

def get_user_pdfs(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, description, upload_date, status, file_size FROM pdf_files WHERE user_id = ? ORDER BY upload_date DESC', (user_id,))
    pdfs = []
    for row in cursor.fetchall():
        pdfs.append({
            "id": row[0], "filename": row[1], "description": row[2],
            "upload_date": row[3], "status": row[4], "file_size": row[5]
        })
    conn.close()
    return pdfs

@app.get("/")
async def root():
    return """
    <html>
    <head><title>Study AI</title></head>
    <body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>🚀 Study AI Backend</h1>
        <p>✅ Pure Python - No Compilation Required</p>
        <p>✅ Gemini AI Integrated</p>
        <p>✅ PDF Processing</p>
        <div style="margin-top: 20px;">
            <a href="/health" style="display: block; margin: 10px;">Health Check</a>
            <a href="/docs" style="display: block; margin: 10px;">API Docs</a>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Study AI - Pure Python",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured",
        "pinecone": "not_configured",
        "features": ["Q&A", "PDF Upload", "Database Storage"]
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(request.question)
        
        return {
            "question": request.question,
            "answer": response.text,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...), description: str = Form("")):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_content = await file.read()
    file_size = len(file_content)
    file_hash = hashlib.md5(file_content).hexdigest()
    
    pdf_id = add_pdf_to_db(file.filename, file_hash, user_id, description, file_size)
    
    if pdf_id is None:
        return {"status": "duplicate", "message": "PDF already uploaded"}
    
    text = extract_text_from_pdf(file_content)
    word_count = len(text.split()) if text else 0
    
    return {
        "status": "success",
        "message": "PDF uploaded successfully",
        "pdf_id": pdf_id,
        "filename": file.filename,
        "word_count": word_count
    }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(user_id: str):
    try:
        pdfs = get_user_pdfs(user_id)
        return {"user_id": user_id, "pdfs": pdfs, "total_pdfs": len(pdfs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/ask-simple")
async def ask_simple(question: str):
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(question)
        return {"question": question, "answer": response.text, "status": "success"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
