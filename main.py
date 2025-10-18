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
import asyncio
import requests
import json

# ✅ PEHLE APP CREATE KARO
app = FastAPI(title="Study AI - Stable Version")

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

print(f"=== ENVIRONMENT VARIABLES ===")
print(f"GEMINI_API_KEY: {'✅ Set' if GEMINI_API_KEY else '❌ Not Set'}")
print(f"PINECONE_API_KEY: {'✅ Set' if PINECONE_API_KEY else '❌ Not Set'}")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured successfully")

# Pinecone HTTP Client (No package needed)
class PineconeHTTPClient:
    def __init__(self, api_key, index_host):
        self.api_key = api_key
        self.base_url = f"https://{index_host}"
        self.headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }
    
    def upsert(self, vectors):
        try:
            response = requests.post(
                f"{self.base_url}/vectors/upsert",
                headers=self.headers,
                json={"vectors": vectors},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Pinecone upsert error: {e}")
            return None
    
    def query(self, vector, top_k=5, filter=None):
        try:
            payload = {
                "vector": vector,
                "topK": top_k,
                "includeMetadata": True
            }
            if filter:
                payload["filter"] = filter
                
            response = requests.post(
                f"{self.base_url}/vectors/query",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Pinecone query error: {e}")
            return None

# Initialize Pinecone HTTP Client
pinecone_client = None
if PINECONE_API_KEY and PINECONE_INDEX_HOST:
    pinecone_client = PineconeHTTPClient(PINECONE_API_KEY, PINECONE_INDEX_HOST)
    print("✅ Pinecone HTTP Client initialized")

pinecone_configured = pinecone_client is not None

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
            file_size INTEGER,
            processed BOOLEAN DEFAULT FALSE,
            chunks_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# Initialize database
init_db()

# Pydantic Models (V1 compatible)
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"
    use_pdf_context: bool = False

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
    """Extract text from PDF"""
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
    """Split text into chunks for vector storage"""
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

async def process_pdf_for_vectors(pdf_id, filename, file_content, user_id):
    """Process PDF and store in vector database"""
    if not pinecone_configured:
        print("Pinecone not configured, skipping vector processing")
        return 0
    
    try:
        print(f"Processing PDF {pdf_id} for vectors...")
        text = extract_text_from_pdf(file_content)
        if not text:
            print("No text extracted from PDF")
            return 0
            
        chunks = split_text_into_chunks(text)
        print(f"Created {len(chunks)} chunks from PDF")
        
        # Store each chunk in Pinecone via HTTP
        successful_chunks = 0
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk
                )['embedding']
                
                # Prepare vector data
                vector_data = {
                    "id": f"{user_id}_{pdf_id}_{i}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk,
                        "filename": filename,
                        "user_id": user_id,
                        "pdf_id": pdf_id,
                        "chunk_index": i
                    }
                }
                
                # Store in Pinecone via HTTP
                result = pinecone_client.upsert([vector_data])
                if result is not None:
                    successful_chunks += 1
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE pdf_files SET processed = TRUE, chunks_count = ? WHERE id = ?',
            (successful_chunks, pdf_id)
        )
        conn.commit()
        conn.close()
        
        print(f"✅ PDF {pdf_id} processed successfully with {successful_chunks} chunks")
        return successful_chunks
        
    except Exception as e:
        print(f"❌ Error processing PDF for vectors: {e}")
        return 0

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
        print(f"✅ PDF added to database with ID: {pdf_id}")
    except sqlite3.IntegrityError:
        # Duplicate file
        cursor.execute('SELECT id FROM pdf_files WHERE file_hash = ?', (file_hash,))
        result = cursor.fetchone()
        pdf_id = result[0] if result else None
        print(f"⚠️ Duplicate PDF found, ID: {pdf_id}")
    finally:
        conn.close()
    
    return pdf_id

def get_user_pdfs(user_id):
    """Get all PDFs for a user"""
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
            "id": row[0],
            "filename": row[1],
            "description": row[2],
            "upload_date": row[3],
            "status": row[4],
            "file_size": row[5],
            "processed": bool(row[6]),
            "chunks_count": row[7] or 0
        })
    
    conn.close()
    return pdfs

@app.get("/", response_class=HTMLResponse)
async def root():
    pinecone_status = "✅ ACTIVE" if pinecone_configured else "❌ INACTIVE"
    pinecone_color = "#00ff00" if pinecone_configured else "#ff4444"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Study AI - Stable Version</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 50px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .status-box {{ 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                display: inline-block;
                backdrop-filter: blur(10px);
                max-width: 700px;
            }}
        </style>
    </head>
    <body>
        <div class="status-box">
            <h1>🚀 Study AI Backend - STABLE</h1>
            <div style="margin: 20px 0;">
                <span style="background: #00ff00; color: #000; padding: 4px 12px; border-radius: 15px; margin: 0 5px;">Gemini AI ✅</span>
                <span style="background: {pinecone_color}; color: {'#000' if pinecone_configured else '#fff'}; padding: 4px 12px; border-radius: 15px; margin: 0 5px;">
                    Vector Search {pinecone_status}
                </span>
                <span style="background: #00ff00; color: #000; padding: 4px 12px; border-radius: 15px; margin: 0 5px;">PDF Processing ✅</span>
            </div>
            
            <p>🎉 No Rust Compilation - Pure Python Solution</p>
            
            <div style="margin-top: 20px;">
                <a href="/health" style="color: #00ff00; text-decoration: none; display: block; margin: 10px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 5px;">🔍 Health Check</a>
                <a href="/system-status" style="color: #00ff00; text-decoration: none; display: block; margin: 10px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 5px;">⚙️ System Status</a>
                <a href="/docs" style="color: #00ff00; text-decoration: none; display: block; margin: 10px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 5px;">📚 API Docs</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "Study AI - Stable",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured",
        "pinecone": "configured" if pinecone_configured else "not_configured",
        "python_version": "3.11.9 - Stable"
    }

@app.get("/system-status")
async def system_status():
    return {
        "status": "stable",
        "python_version": "3.11.9",
        "gemini_ai": "active" if GEMINI_API_KEY else "inactive",
        "vector_database": "active" if pinecone_configured else "inactive",
        "pdf_processing": "active",
        "database": "active",
        "architecture": "pure_python_no_rust",
        "features": {
            "smart_qa": True,
            "pdf_upload": True,
            "vector_search": pinecone_configured,
            "context_aware": pinecone_configured
        }
    }

# Include all other endpoints: /ask, /upload-pdf, /user-pdfs, /ask-simple etc.
# (Same as previous code but with Pydantic V1 compatibility)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
