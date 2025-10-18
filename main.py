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
import re

app = FastAPI(title="Study AI - Simple PDF Q&A")

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
            file_size INTEGER,
            processed BOOLEAN DEFAULT FALSE,
            text_content TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"
    use_pdf_context: bool = True

class PDFInfo(BaseModel):
    id: int
    filename: str
    description: str
    upload_date: str
    status: str
    file_size: int
    processed: bool

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

def find_relevant_text_simple(question, user_id):
    """Simple keyword-based search for relevant text"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all processed PDFs for user
    cursor.execute('''
        SELECT text_content, filename 
        FROM pdf_files 
        WHERE user_id = ? AND processed = TRUE AND text_content IS NOT NULL
    ''', (user_id,))
    
    pdfs = cursor.fetchall()
    conn.close()
    
    if not pdfs:
        return []
    
    # Simple keyword matching
    question_keywords = set(re.findall(r'\w+', question.lower()))
    relevant_sections = []
    
    for text_content, filename in pdfs:
        if not text_content:
            continue
            
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text_content)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            sentence_keywords = set(re.findall(r'\w+', sentence.lower()))
            common_keywords = question_keywords.intersection(sentence_keywords)
            
            # If enough keywords match, consider it relevant
            if len(common_keywords) >= 2:  # At least 2 common keywords
                relevance_score = len(common_keywords) / len(question_keywords)
                relevant_sections.append({
                    'text': sentence.strip(),
                    'filename': filename,
                    'score': relevance_score
                })
    
    # Sort by relevance and return top 3
    relevant_sections.sort(key=lambda x: x['score'], reverse=True)
    return relevant_sections[:3]

def process_pdf_content(pdf_id, text_content):
    """Store PDF text content"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE pdf_files SET processed = TRUE, text_content = ? WHERE id = ?',
            (text_content, pdf_id)
        )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error storing PDF content: {e}")
        return False

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
    cursor.execute('''
        SELECT id, filename, description, upload_date, status, file_size, processed
        FROM pdf_files 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (user_id,))
    
    pdfs = []
    for row in cursor.fetchall():
        pdfs.append({
            "id": row[0], "filename": row[1], "description": row[2],
            "upload_date": row[3], "status": row[4], "file_size": row[5],
            "processed": bool(row[6])
        })
    conn.close()
    return pdfs

@app.get("/")
async def root():
    return {"status": "active", "service": "Study AI - Simple PDF Q&A"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Study AI - Simple PDF Q&A",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured",
        "features": ["Smart Q&A", "PDF Context Processing", "Keyword Search"]
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        context = ""
        sources_used = 0
        
        # If PDF context is enabled, find relevant content
        if request.use_pdf_context:
            relevant_sections = find_relevant_text_simple(request.question, request.user_id)
            
            if relevant_sections:
                context_chunks = []
                for section in relevant_sections:
                    context_chunks.append(f"From {section['filename']}: {section['text']}")
                
                context = "\n\nRelevant information from your PDFs:\n" + "\n---\n".join(context_chunks)
                sources_used = len(relevant_sections)
        
        # Create smart prompt with context
        if context:
            prompt = f"""
            You are a helpful study assistant. Use the provided context from the user's PDF documents to answer their question accurately.

            QUESTION: {request.question}

            {context}

            Please provide a comprehensive answer that:
            1. Directly addresses the question
            2. Uses relevant information from the provided context when applicable
            3. Explains concepts clearly for better understanding
            4. If the context doesn't contain relevant information, use your general knowledge but mention this

            ANSWER:
            """
        else:
            prompt = f"""
            You are a helpful study assistant. Answer the following question clearly and concisely.

            QUESTION: {request.question}

            Provide a comprehensive answer that would help a student understand the concept.

            ANSWER:
            """
        
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)
        
        return {
            "question": request.question,
            "answer": response.text,
            "status": "success",
            "context_used": sources_used > 0,
            "sources_used": sources_used,
            "search_method": "keyword_based"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

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
    
    # Extract and process text
    text_content = extract_text_from_pdf(file_content)
    word_count = len(text_content.split()) if text_content else 0
    
    # Store PDF content for context search
    processed = False
    if text_content:
        processed = process_pdf_content(pdf_id, text_content)
    
    return {
        "status": "success",
        "message": "PDF uploaded successfully",
        "pdf_id": pdf_id,
        "filename": file.filename,
        "word_count": word_count,
        "description": description,
        "processed": processed,
        "context_search_available": True
    }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(user_id: str):
    try:
        pdfs = get_user_pdfs(user_id)
        return {
            "user_id": user_id,
            "pdfs": pdfs,
            "total_pdfs": len(pdfs),
            "processed_pdfs": sum(1 for p in pdfs if p['processed'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/ask-simple")
async def ask_simple(question: str, user_id: str = "default", use_context: bool = True):
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    try:
        request = QuestionRequest(question=question, user_id=user_id, use_pdf_context=use_context)
        return await ask_question(request)
    except Exception as e:
        return {"error": str(e)}

@app.delete("/delete-pdf/{pdf_id}")
async def delete_pdf(pdf_id: int, user_id: str):
    """Delete a PDF and its content"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if PDF exists and belongs to user
        cursor.execute('SELECT id FROM pdf_files WHERE id = ? AND user_id = ?', (pdf_id, user_id))
        pdf_info = cursor.fetchone()
        
        if not pdf_info:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Delete from database
        cursor.execute('DELETE FROM pdf_files WHERE id = ?', (pdf_id,))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "PDF deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
