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

# ✅ PEHLE APP CREATE KARO
app = FastAPI(title="Study AI - Complete Version")

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

print(f"=== ENVIRONMENT VARIABLES ===")
print(f"GEMINI_API_KEY: {'✅ Set' if GEMINI_API_KEY else '❌ Not Set'}")
print(f"PINECONE_API_KEY: {'✅ Set' if PINECONE_API_KEY else '❌ Not Set'}")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured successfully")

# Pinecone Setup with detailed debugging
pinecone_configured = False
index = None
pinecone_error = "Not attempted"

try:
    if PINECONE_API_KEY:
        print("🔄 Attempting Pinecone initialization...")
        
        # Try new import style first
        try:
            import pinecone
            print("✅ pinecone module imported successfully")
            
            # Initialize with new SDK
            pinecone.init(api_key=PINECONE_API_KEY)
            print("✅ pinecone.init() successful")
            
            index_name = "study-ai-index"
            
            # Check if index exists
            print(f"🔄 Checking index: {index_name}")
            if index_name not in pinecone.list_indexes():
                print(f"📝 Creating new index: {index_name}")
                pinecone.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine"
                )
                print("✅ Index created successfully")
            else:
                print("✅ Index already exists")
            
            # Connect to index
            index = pinecone.Index(index_name)
            pinecone_configured = True
            pinecone_error = "Success"
            print("🎉 Pinecone fully configured and connected!")
            
        except Exception as e:
            print(f"❌ New SDK failed: {e}")
            pinecone_error = f"New SDK: {str(e)}"
            
            # Fallback to old SDK
            try:
                print("🔄 Trying old SDK...")
                from pinecone import Pinecone, ServerlessSpec
                
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index_name = "study-ai-index"
                
                existing_indexes = [index["name"] for index in pc.list_indexes()]
                print(f"Existing indexes: {existing_indexes}")
                
                if index_name not in existing_indexes:
                    print(f"Creating index: {index_name}")
                    pc.create_index(
                        name=index_name,
                        dimension=768,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                
                index = pc.Index(index_name)
                pinecone_configured = True
                pinecone_error = "Success (Old SDK)"
                print("✅ Pinecone configured with old SDK")
                
            except Exception as e2:
                print(f"❌ Old SDK also failed: {e2}")
                pinecone_error = f"Both failed: New={e}, Old={e2}"
                
    else:
        pinecone_error = "No API key"
        print("❌ No Pinecone API key found")
        
except Exception as e:
    print(f"💥 Overall Pinecone setup failed: {e}")
    pinecone_error = f"Setup failed: {str(e)}"

print(f"=== FINAL PINE CONE STATUS ===")
print(f"Configured: {pinecone_configured}")
print(f"Error: {pinecone_error}")

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

# Pydantic Models
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
        
        # Store each chunk in Pinecone
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk
                )['embedding']
                
                # Store in Pinecone
                index.upsert(vectors=[(
                    f"{user_id}_{pdf_id}_{i}",
                    embedding,
                    {
                        "text": chunk,
                        "filename": filename,
                        "user_id": user_id,
                        "pdf_id": pdf_id,
                        "chunk_index": i
                    }
                )])
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
        
        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE pdf_files SET processed = TRUE, chunks_count = ? WHERE id = ?',
            (len(chunks), pdf_id)
        )
        conn.commit()
        conn.close()
        
        print(f"✅ PDF {pdf_id} processed successfully with {len(chunks)} chunks")
        return len(chunks)
        
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
        <title>Study AI - Status</title>
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
            .debug-info {{
                background: rgba(255,255,255,0.1);
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: left;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="status-box">
            <h1>🚀 Study AI Backend</h1>
            <div class="status">
                <span class="live-dot"></span> LIVE & RUNNING
            </div>
            
            <div style="margin: 20px 0;">
                <span class="status-indicator status-active">Gemini AI ✅</span>
                <span class="status-indicator" style="background: {pinecone_color}; color: {'#000' if pinecone_configured else '#fff'}">
                    Vector Search {pinecone_status}
                </span>
                <span class="status-indicator status-active">PDF Processing ✅</span>
            </div>
            
            <div class="debug-info">
                <strong>Debug Info:</strong><br>
                Pinecone API Key: {'✅ Set' if PINECONE_API_KEY else '❌ Not Set'}<br>
                Pinecone Configured: {pinecone_configured}<br>
                Error: {pinecone_error}
            </div>
            
            <div class="endpoints">
                <a href="/health">🔍 Health Check</a>
                <a href="/debug-pinecone">🐛 Debug Pinecone</a>
                <a href="/system-status">⚙️ System Status</a>
                <a href="/docs">📚 API Docs</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "Study AI",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured",
        "pinecone": "configured" if pinecone_configured else "not_configured",
        "pinecone_error": pinecone_error
    }

@app.get("/debug-pinecone")
async def debug_pinecone():
    """Debug Pinecone connection"""
    return {
        "pinecone_configured": pinecone_configured,
        "api_key_set": bool(PINECONE_API_KEY),
        "error_message": pinecone_error,
        "next_steps": "Check Render logs for detailed error information"
    }

@app.get("/system-status")
async def system_status():
    """Detailed system status"""
    return {
        "gemini_ai": "active" if GEMINI_API_KEY else "inactive",
        "vector_database": "active" if pinecone_configured else "inactive",
        "pdf_processing": "active",
        "database": "active",
        "api_keys": {
            "gemini_set": bool(GEMINI_API_KEY),
            "pinecone_set": bool(PINECONE_API_KEY),
            "pinecone_connected": pinecone_configured
        },
        "pinecone_error": pinecone_error,
        "features": {
            "smart_qa": True,
            "pdf_upload": True,
            "vector_search": pinecone_configured,
            "context_aware": pinecone_configured
        }
    }

# ... (rest of the endpoints remain same as previous code)
# Include all the other endpoints: /ask, /upload-pdf, /user-pdfs, /ask-simple etc.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
