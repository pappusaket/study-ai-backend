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

app = FastAPI(
    title="Study-AI Backend",
    description="Backend API for Study-AI application",
    version="0.1.0"
)

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

class QuizRequest(BaseModel):
    user_id: str

class QuizResponse(BaseModel):
    question: str
    options: list
    correct_answer: str
    explanation: str
    user_level: str = "beginner"

@app.get("/")
async def health_check():
    return {"message": "Welcome to the Study-AI Backend!"}

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
                answer="मुझे आपके सवाल का जवाब देने के लिए पर्याप्त जानकारी नहीं मिली।",
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

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """PDF अपलोड करें और process करें"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="सिर्फ PDF files allowed हैं")
    
    try:
        contents = await file.read()
        
        # PDF पढ़ें
        pdf_reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF से कोई टेक्स्ट नहीं मिला")
        
        # Simple processing for now
        return {
            "success": True,
            "message": f"PDF {file.filename} received successfully!",
            "text_preview": text[:200] + "..."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
