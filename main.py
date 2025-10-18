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
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Study AI - Smart PDF Q&A")

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

# Load sentence transformer model for semantic search
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_MODEL_LOADED = True
except:
    EMBEDDING_MODEL_LOADED = False

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
            text_content TEXT,
            chunks_count INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_id INTEGER,
            chunk_index INTEGER,
            chunk_text TEXT,
            embedding BLOB,
            FOREIGN KEY (pdf_id) REFERENCES pdf_files (id)
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
    chunks_count: int

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

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into chunks for semantic search"""
    if not text:
        return []
        
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    
    return chunks

def get_text_embedding(text):
    """Get embedding for text using sentence transformers"""
    if not EMBEDDING_MODEL_LOADED:
        return None
    
    try:
        embedding = model.encode([text])[0]
        return embedding.tobytes()  # Store as bytes in database
    except:
        return None

def find_similar_chunks(question, user_id, top_k=3):
    """Find most relevant PDF chunks for a question"""
    if not EMBEDDING_MODEL_LOADED:
        return []
    
    try:
        # Get question embedding
        question_embedding = model.encode([question])[0]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all chunks for user
        cursor.execute('''
            SELECT pc.chunk_text, pc.embedding, pf.filename
            FROM pdf_chunks pc
            JOIN pdf_files pf ON pc.pdf_id = pf.id
            WHERE pf.user_id = ? AND pf.processed = TRUE
        ''', (user_id,))
        
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            return []
        
        # Calculate similarities
        similarities = []
        for chunk_text, embedding_bytes, filename in chunks:
            if embedding_bytes:
                chunk_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][0]
                similarities.append((chunk_text, similarity, filename))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []

def process_pdf_content(pdf_id, text_content, user_id):
    """Process PDF text and store chunks with embeddings"""
    if not EMBEDDING_MODEL_LOADED:
        return 0
    
    try:
        chunks = split_text_into_chunks(text_content)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Store chunks with embeddings
        chunks_stored = 0
        for i, chunk in enumerate(chunks):
            embedding = get_text_embedding(chunk)
            if embedding:
                cursor.execute('''
                    INSERT INTO pdf_chunks (pdf_id, chunk_index, chunk_text, embedding)
                    VALUES (?, ?, ?, ?)
                ''', (pdf_id, i, chunk, embedding))
                chunks_stored += 1
        
        # Update PDF status
        cursor.execute(
            'UPDATE pdf_files SET processed = TRUE, chunks_count = ?, text_content = ? WHERE id = ?',
            (chunks_stored, text_content, pdf_id)
        )
        
        conn.commit()
        conn.close()
        return chunks_stored
        
    except Exception as e:
        print(f"Error processing PDF content: {e}")
        return 0

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
        SELECT id, filename, description, upload_date, status, file_size, processed, chunks_count
        FROM pdf_files 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (user_id,))
    
    pdfs = []
    for row in cursor.fetchall():
        pdfs.append({
            "id": row[0], "filename": row[1], "description": row[2],
            "upload_date": row[3], "status": row[4], "file_size": row[5],
            "processed": bool(row[6]), "chunks_count": row[7] or 0
        })
    conn.close()
    return pdfs

@app.get("/")
async def root():
    return {"status": "active", "service": "Study AI - Smart PDF Q&A"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Study AI - Smart PDF Q&A",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured",
        "semantic_search": "available" if EMBEDDING_MODEL_LOADED else "unavailable",
        "features": ["Smart Q&A", "PDF Context Processing", "Semantic Search"]
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        context = ""
        sources_used = 0
        
        # If PDF context is enabled, find relevant content
        if request.use_pdf_context and EMBEDDING_MODEL_LOADED:
            similar_chunks = find_similar_chunks(request.question, request.user_id)
            
            if similar_chunks:
                context_chunks = []
                for chunk_text, similarity, filename in similar_chunks:
                    context_chunks.append(f"From {filename} (relevance: {similarity:.2f}): {chunk_text}")
                
                context = "\n\nRelevant information from your PDFs:\n" + "\n---\n".join(context_chunks)
                sources_used = len(similar_chunks)
        
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
            "semantic_search": EMBEDDING_MODEL_LOADED
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
    
    # Process PDF for semantic search (in background)
    chunks_processed = 0
    if EMBEDDING_MODEL_LOADED and text_content:
        chunks_processed = process_pdf_content(pdf_id, text_content, user_id)
    
    return {
        "status": "success",
        "message": "PDF uploaded successfully",
        "pdf_id": pdf_id,
        "filename": file.filename,
        "word_count": word_count,
        "description": description,
        "processed": chunks_processed > 0,
        "chunks_processed": chunks_processed,
        "semantic_search_available": EMBEDDING_MODEL_LOADED
    }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(user_id: str):
    try:
        pdfs = get_user_pdfs(user_id)
        return {
            "user_id": user_id,
            "pdfs": pdfs,
            "total_pdfs": len(pdfs),
            "processed_pdfs": sum(1 for p in pdfs if p['processed']),
            "semantic_search": EMBEDDING_MODEL_LOADED
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
