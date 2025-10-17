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
import traceback

# ✅ PEHLE APP CREATE KARO
app = FastAPI(title="Study AI - Debug Version")

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

print("=" * 50)
print("ENVIRONMENT VARIABLES CHECK:")
print(f"GEMINI_API_KEY: {'✅ SET' if GEMINI_API_KEY else '❌ NOT SET'}")
print(f"PINECONE_API_KEY: {'✅ SET' if PINECONE_API_KEY else '❌ NOT SET'}")
print("=" * 50)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Pinecone Setup with detailed debugging
pinecone_configured = False
index = None
pinecone_error = "Not attempted"

try:
    if PINECONE_API_KEY:
        print("🔄 Attempting Pinecone initialization...")
        
        # Try different import approaches
        try:
            from pinecone import Pinecone, ServerlessSpec
            print("✅ Pinecone import successful")
        except ImportError as e:
            print(f"❌ Pinecone import failed: {e}")
            raise e
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Pinecone client created")
        
        index_name = "study-ai-index"
        
        # Check if index exists
        try:
            existing_indexes = [index["name"] for index in pc.list_indexes()]
            print(f"✅ Existing indexes: {existing_indexes}")
        except Exception as e:
            print(f"❌ Failed to list indexes: {e}")
            raise e
        
        if index_name not in existing_indexes:
            print(f"🔄 Creating index: {index_name}")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print("✅ Index created successfully")
            except Exception as e:
                print(f"❌ Index creation failed: {e}")
                raise e
        else:
            print("✅ Index already exists")
        
        # Connect to index
        try:
            index = pc.Index(index_name)
            print("✅ Index connection successful")
            
            # Test the connection
            stats = index.describe_index_stats()
            print(f"✅ Index stats: {stats}")
            
            pinecone_configured = True
            pinecone_error = "None"
            print("🎉 Pinecone fully configured and connected!")
            
        except Exception as e:
            print(f"❌ Index connection failed: {e}")
            pinecone_error = f"Index connection: {str(e)}"
            raise e
        
except Exception as e:
    print(f"💥 Pinecone setup failed: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    pinecone_error = str(e)
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

@app.get("/", response_class=HTMLResponse)
async def root():
    pinecone_status = "✅ ACTIVE" if pinecone_configured else "❌ INACTIVE"
    pinecone_color = "#00ff00" if pinecone_configured else "#ff4444"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Study AI - Debug</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .status-box {{ 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                display: inline-block;
                backdrop-filter: blur(10px);
                max-width: 800px;
                text-align: left;
            }}
            .status-item {{
                margin: 10px 0;
                padding: 10px;
                background: rgba(0,0,0,0.2);
                border-radius: 5px;
            }}
            .success {{ color: #00ff00; }}
            .error {{ color: #ff4444; }}
            .warning {{ color: #ffaa00; }}
        </style>
    </head>
    <body>
        <div class="status-box">
            <h1>🔧 Study AI - Debug Mode</h1>
            
            <div class="status-item">
                <strong>Gemini AI:</strong> 
                <span class="{'success' if GEMINI_API_KEY else 'error'}">
                    {'✅ CONFIGURED' if GEMINI_API_KEY else '❌ NOT CONFIGURED'}
                </span>
            </div>
            
            <div class="status-item">
                <strong>Pinecone Vector DB:</strong> 
                <span class="{'success' if pinecone_configured else 'error'}">
                    {'✅ ACTIVE' if pinecone_configured else '❌ INACTIVE'}
                </span>
            </div>
            
            <div class="status-item">
                <strong>Pinecone API Key:</strong> 
                <span class="{'success' if PINECONE_API_KEY else 'error'}">
                    {'✅ SET' if PINECONE_API_KEY else '❌ NOT SET'}
                </span>
            </div>
            
            {f'<div class="status-item error"><strong>Pinecone Error:</strong> {pinecone_error}</div>' if not pinecone_configured else ''}
            
            <div class="status-item">
                <strong>Endpoints:</strong>
                <ul>
                    <li><a href="/health" style="color: #00ff00;">/health</a> - Basic health check</li>
                    <li><a href="/debug-pinecone" style="color: #00ff00;">/debug-pinecone</a> - Pinecone debug</li>
                    <li><a href="/system-status" style="color: #00ff00;">/system-status</a> - System status</li>
                    <li><a href="/docs" style="color: #00ff00;">/docs</a> - API documentation</li>
                </ul>
            </div>
            
            <div class="status-item warning">
                <strong>Note:</strong> Check Render logs for detailed initialization logs
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "Study AI - Debug",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured",
        "pinecone": "configured" if pinecone_configured else "not_configured",
        "pinecone_error": pinecone_error if not pinecone_configured else "none"
    }

@app.get("/debug-pinecone")
async def debug_pinecone():
    """Detailed Pinecone debug information"""
    return {
        "pinecone_configured": pinecone_configured,
        "api_key_set": bool(PINECONE_API_KEY),
        "error_message": pinecone_error,
        "environment_check": {
            "gemini_key": "set" if GEMINI_API_KEY else "not_set",
            "pinecone_key": "set" if PINECONE_API_KEY else "not_set"
        }
    }

@app.get("/system-status")
async def system_status():
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

# Basic Q&A endpoint (without vector search for now)
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
            "status": "success",
            "vector_search": "inactive",
            "message": "Basic Q&A working (vector search inactive)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
