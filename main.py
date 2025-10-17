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
from typing import Optional

# ✅ PEHLE APP CREATE KARO
app = FastAPI(title="Study AI - With PDF Support")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Database setup
DB_PATH = "study_ai.db"

def init_db():
    """Initialize SQLite database"""
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

# Initialize database
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
    """Extract text from PDF"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF reading error: {str(e)}")

def add_pdf_to_db(filename, file_hash, user_id, description, file_size):
    """Add PDF info to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO pdf_files 
            (filename, file_hash, user_id, description, upload_date, status, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename, file_hash, user_id, description, 
            datetime.now().isoformat(), 'uploaded', file_size
        ))
        
        conn.commit()
        pdf_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # Duplicate file
        cursor.execute('SELECT id FROM pdf_files WHERE file_hash = ?', (file_hash,))
        result = cursor.fetchone()
        pdf_id = result[0] if result else None
    finally:
        conn.close()
    
    return pdf_id

def get_user_pdfs(user_id):
    """Get all PDFs for a user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, description, upload_date, status, file_size
        FROM pdf_files 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (user_id,))
    
    pdfs = []
    for row in cursor.fetchall():
        pdfs.append({
            "id": row[0],
            "filename": row[1],
            "description": row[2],
            "upload_date": row[3],
            "status": row[4],
            "file_size": row[5]
        })
    
    conn.close()
    return pdfs

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Study AI - Status</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 50px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .status-box { 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                display: inline-block;
                backdrop-filter: blur(10px);
                max-width: 600px;
            }
            .status { 
                font-size: 24px; 
                margin: 20px 0; 
            }
            .live-dot {
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #00ff00;
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 10px;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .endpoints {
                margin-top: 20px;
            }
            a {
                color: #00ff00;
                text-decoration: none;
                display: block;
                margin: 10px;
                padding: 10px;
                background: rgba(0,0,0,0.3);
                border-radius: 5px;
                transition: all 0.3s;
            }
            a:hover {
                background: rgba(0,0,0,0.5);
                transform: translateY(-2px);
            }
            .feature-badge {
                background: #00ff00;
                color: #333;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
            .feature-list {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="status-box">
            <h1>🚀 Study AI Backend</h1>
            <div class="status">
                <span class="live-dot"></span> LIVE & RUNNING
            </div>
            <p>Smart Q&A System with PDF Upload Support</p>
            
            <div class="endpoints">
                <a href="/health">🔍 Health Check</a>
                <a href="/docs">📚 API Documentation <span class="feature-badge">NEW</span></a>
                <a href="/test">🧪 Test Endpoint</a>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                <h3>🎯 Available Features</h3>
                <div class="feature-list">
                    <p>✅ Smart Q&A System</p>
                    <p>✅ PDF Upload & Storage</p>
                    <p>✅ Multi-language Support</p>
                    <p>✅ Fast Response Time</p>
                    <p>✅ Database Storage</p>
                    <p>✅ File Duplicate Check</p>
                </div>
                <p style="font-size: 12px; opacity: 0.8; margin-top: 15px;">
                    Use /upload-pdf endpoint to upload study materials
                </p>
            </div>
            
            <div style="margin-top: 20px; font-size: 12px; opacity: 0.8;">
                Last checked: <span id="timestamp">loading...</span>
            </div>
        </div>

        <script>
            function updateTime() {
                document.getElementById('timestamp').textContent = new Date().toLocaleString();
            }
            updateTime();
            setInterval(updateTime, 1000);
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    gemini_status = "configured" if GEMINI_API_KEY else "not_configured"
    return {
        "status": "healthy", 
        "service": "Study AI",
        "gemini": gemini_status,
        "features": ["Q&A System", "PDF Upload", "Multi-language", "Database Storage"]
    }

@app.get("/test")
async def test():
    return {"test": "passed", "message": "Everything is working!"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask questions and get AI-powered answers"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        prompt = f"""
        You are a helpful study assistant. Answer the following question clearly and concisely.
        
        QUESTION: {request.question}
        
        Provide a comprehensive answer that would help a student understand the concept.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return {
            "question": request.question,
            "answer": response.text,
            "status": "success",
            "user_id": request.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    description: str = Form("")
):
    """Upload PDF for processing"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_content = await file.read()
    file_size = len(file_content)
    
    # Check duplicate
    file_hash = hashlib.md5(file_content).hexdigest()
    
    # Add to database
    pdf_id = add_pdf_to_db(
        filename=file.filename,
        file_hash=file_hash,
        user_id=user_id,
        description=description,
        file_size=file_size
    )
    
    if pdf_id is None:
        return {
            "status": "duplicate", 
            "message": "This PDF has already been uploaded",
            "pdf_id": None
        }
    
    # Extract text (basic processing)
    try:
        text = extract_text_from_pdf(file_content)
        word_count = len(text.split())
        
        return {
            "status": "success",
            "message": "PDF uploaded successfully",
            "pdf_id": pdf_id,
            "filename": file.filename,
            "file_size": file_size,
            "word_count": word_count,
            "description": description
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing PDF: {str(e)}",
            "pdf_id": pdf_id
        }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(user_id: str):
    """Get all PDFs for a user"""
    try:
        pdfs = get_user_pdfs(user_id)
        return {
            "user_id": user_id,
            "pdfs": pdfs,
            "total_pdfs": len(pdfs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching PDFs: {str(e)}")

@app.get("/ask-simple")
async def ask_simple(question: str):
    """Quick test endpoint for questions"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(question)
        
        return {
            "question": question,
            "answer": response.text,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
