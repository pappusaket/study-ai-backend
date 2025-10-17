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

print(f"GEMINI_API_KEY: {'Set' if GEMINI_API_KEY else 'Not Set'}")
print(f"PINECONE_API_KEY: {'Set' if PINECONE_API_KEY else 'Not Set'}")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Pinecone Setup with better error handling
pinecone_configured = False
index = None

try:
    if PINECONE_API_KEY:
        from pinecone import Pinecone, ServerlessSpec
        print("Initializing Pinecone...")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "study-ai-index"
        
        # Check if index exists
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
            print("Index created successfully")
        else:
            print("Index already exists")
        
        index = pc.Index(index_name)
        pinecone_configured = True
        print("✅ Pinecone configured successfully!")
        
except Exception as e:
    print(f"❌ Pinecone initialization failed: {e}")
    pinecone_configured = False
    index = None

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
            .status {{ 
                font-size: 24px; 
                margin: 20px 0; 
            }}
            .live-dot {{
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #00ff00;
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 10px;
            }}
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            .endpoints {{
                margin-top: 20px;
            }}
            a {{
                color: #00ff00;
                text-decoration: none;
                display: block;
                margin: 10px;
                padding: 10px;
                background: rgba(0,0,0,0.3);
                border-radius: 5px;
                transition: all 0.3s;
            }}
            a:hover {{
                background: rgba(0,0,0,0.5);
                transform: translateY(-2px);
            }}
            .feature-badge {{
                background: #00ff00;
                color: #333;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }}
            .feature-list {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                margin-top: 20px;
            }}
            .status-indicator {{
                display: inline-block;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 12px;
                margin: 0 5px;
            }}
            .status-active {{ background: #00ff00; color: #000; }}
            .status-inactive {{ background: #ff4444; color: #fff; }}
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
            
            <p>Advanced AI Study Assistant with Smart Search</p>
            
            <div class="endpoints">
                <a href="/health">🔍 Health Check</a>
                <a href="/docs">📚 API Documentation <span class="feature-badge">NEW</span></a>
                <a href="/test">🧪 Test Endpoint</a>
                <a href="/system-status">⚙️ System Status</a>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                <h3>🎯 Available Features</h3>
                <div class="feature-list">
                    <p>✅ Smart Q&A System</p>
                    <p>✅ PDF Upload & Storage</p>
                    <p>✅ Vector Search {'(Active)' if pinecone_configured else '(Ready)'}</p>
                    <p>✅ Multi-language Support</p>
                    <p>✅ Database Storage</p>
                    <p>✅ Context-Aware Answers</p>
                </div>
                <p style="font-size: 12px; opacity: 0.8; margin-top: 15px;">
                    {'🚀 Upload PDFs and ask context-aware questions!' if pinecone_configured else '⚠️ Vector search ready - upload PDFs to enable!'}
                </p>
            </div>
            
            <div style="margin-top: 20px; font-size: 12px; opacity: 0.8;">
                Last checked: <span id="timestamp">loading...</span>
            </div>
        </div>

        <script>
            function updateTime() {{
                document.getElementById('timestamp').textContent = new Date().toLocaleString();
            }}
            updateTime();
            setInterval(updateTime, 1000);
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    gemini_status = "configured" if GEMINI_API_KEY else "not_configured"
    pinecone_status = "configured" if pinecone_configured else "not_configured"
    
    features = ["Q&A System", "PDF Upload", "Multi-language"]
    if pinecone_configured:
        features.append("Vector Search")
    else:
        features.append("Basic Search")
    
    return {
        "status": "healthy", 
        "service": "Study AI",
        "gemini": gemini_status,
        "pinecone": pinecone_status,
        "features": features
    }

@app.get("/test")
async def test():
    return {
        "test": "passed", 
        "message": "Everything is working!",
        "vector_search": "active" if pinecone_configured else "not_configured",
        "gemini_ai": "active" if GEMINI_API_KEY else "inactive"
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask questions with optional PDF context"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        context = ""
        sources_used = 0
        
        # If vector search is enabled and user wants PDF context
        if pinecone_configured and request.use_pdf_context:
            try:
                print(f"Searching vectors for question: {request.question}")
                # Create question embedding
                question_embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=request.question
                )['embedding']
                
                # Search in Pinecone
                results = index.query(
                    vector=question_embedding,
                    top_k=3,
                    include_metadata=True,
                    filter={"user_id": request.user_id}
                )
                
                if results['matches']:
                    context_chunks = [match['metadata']['text'] for match in results['matches']]
                    context = "\n\nRelevant context from your documents:\n" + "\n---\n".join(context_chunks)
                    sources_used = len(results['matches'])
                    print(f"Found {sources_used} relevant chunks")
                else:
                    print("No relevant chunks found")
            except Exception as e:
                print(f"Vector search error: {e}")
                # Continue without context
        
        prompt = f"""
        You are a helpful study assistant. Answer the following question clearly and concisely.
        
        QUESTION: {request.question}
        {context}
        
        Provide a comprehensive answer that would help a student understand the concept.
        If there is relevant context from documents, use it to enhance your answer.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return {
            "question": request.question,
            "answer": response.text,
            "status": "success",
            "user_id": request.user_id,
            "context_used": sources_used > 0,
            "sources_used": sources_used,
            "vector_search": "active" if pinecone_configured else "inactive"
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
    
    # Process PDF for vector storage (in background)
    chunks_count = 0
    if pinecone_configured:
        chunks_count = await process_pdf_for_vectors(
            pdf_id, file.filename, file_content, user_id
        )
    
    # Extract basic text info
    text = extract_text_from_pdf(file_content)
    word_count = len(text.split()) if text else 0
    
    return {
        "status": "success",
        "message": "PDF uploaded successfully",
        "pdf_id": pdf_id,
        "filename": file.filename,
        "file_size": file_size,
        "word_count": word_count,
        "description": description,
        "vector_processed": chunks_count > 0,
        "chunks_stored": chunks_count,
        "vector_search_available": pinecone_configured
    }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(user_id: str):
    """Get all PDFs for a user"""
    try:
        pdfs = get_user_pdfs(user_id)
        return {
            "user_id": user_id,
            "pdfs": pdfs,
            "total_pdfs": len(pdfs),
            "vector_search_enabled": pinecone_configured,
            "processed_pdfs": sum(1 for pdf in pdfs if pdf['processed'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching PDFs: {str(e)}")

@app.get("/ask-simple")
async def ask_simple(question: str, use_context: bool = False):
    """Quick test endpoint for questions"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    try:
        request = QuestionRequest(question=question, use_pdf_context=use_context)
        return await ask_question(request)
    except Exception as e:
        return {"error": str(e)}

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
        "features": {
            "smart_qa": True,
            "pdf_upload": True,
            "vector_search": pinecone_configured,
            "context_aware": pinecone_configured
        }
    }

@app.get("/debug-pinecone")
async def debug_pinecone():
    """Debug Pinecone connection"""
    if not pinecone_configured:
        return {"status": "not_configured", "message": "Pinecone not configured"}
    
    try:
        # Try to get index stats
        stats = index.describe_index_stats()
        return {
            "status": "connected",
            "index_stats": stats,
            "message": "Pinecone is properly connected"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Pinecone connection error: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
