import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
import io
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Study AI Backend", version="1.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="study_documents")
    logger.info("AI models initialized successfully")
except Exception as e:
    logger.error(f"Initialization error: {e}")

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "AI_KEY_NOT_SET":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini AI configured successfully")
    except Exception as e:
        logger.error(f"Gemini configuration error: {e}")
        gemini_model = None
else:
    logger.warning("GEMINI_API_KEY not set")
    gemini_model = None

# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    user_id: str = "default"
    language: str = "hindi"

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    success: bool

def extract_text_from_pdf(pdf_file):
    """PDF से टेक्स्ट निकालें"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF पढ़ने में error: {str(e)}")

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """टेक्स्ट को छोटे chunks में तोड़ें"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    
    return chunks

@app.get("/")
async def health_check():
    return {
        "status": "active", 
        "message": "Study AI Backend is running!",
        "version": "1.0",
        "ai_ready": gemini_model is not None
    }

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """PDF अपलोड करें और process करें"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="सिर्फ PDF files allowed हैं")
    
    try:
        # PDF पढ़ें
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF से कोई टेक्स्ट नहीं मिला")
        
        # Chunks बनाएं
        chunks = split_text_into_chunks(text)
        
        # Embeddings बनाएं और ChromaDB में store करें
        embeddings = embedding_model.encode(chunks).tolist()
        
        # Unique IDs generate करें
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = [{"filename": file.filename, "chunk_index": i} for i in range(len(chunks))]
        
        # ChromaDB में add करें
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "success": True,
            "message": f"PDF successfully processed! {len(chunks)} chunks added.",
            "filename": file.filename,
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """सवाल पूछें और जवाब पाएं"""
    try:
        # सवाल का embedding बनाएं
        question_embedding = embedding_model.encode([request.question]).tolist()[0]
        
        # Similar chunks खोजें
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        if not results['documents'] or not results['documents'][0]:
            return QueryResponse(
                answer="मुझे आपके सवाल का जवाब देने के लिए पर्याप्त जानकारी नहीं मिली। कृपया PDFs upload करें।",
                sources=[],
                success=False
            )
        
        # Context बनाएं
        context = "\n\n".join(results['documents'][0])
        
        if not gemini_model:
            return QueryResponse(
                answer="AI service is currently unavailable. Please configure GEMINI_API_KEY.",
                sources=[],
                success=False
            )
        
        # Prompt template
        prompt = f"""
        CONTEXT FROM STUDY MATERIALS:
        {context}

        USER QUESTION: {request.question}

        You are a helpful teaching assistant. Follow these rules:

        1. Answer using ONLY the information from CONTEXT
        2. If answer is not in context, say "यह जानकारी मेरे पास उपलब्ध नहीं है"
        3. Explain in simple, clear {request.language}
        4. Use bullet points for key information  
        5. Add examples if available in context
        6. Keep answers focused and relevant

        ANSWER IN {request.language.upper()}:
        """
        
        response = gemini_model.generate_content(prompt)
        
        return QueryResponse(
            answer=response.text,
            sources=[f"Source {i+1}" for i in range(len(results['documents'][0]))],
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        return QueryResponse(
            answer=f"Error processing your question: {str(e)}",
            sources=[],
            success=False
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
