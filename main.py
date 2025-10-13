import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pypdf import PdfReader
import io
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Study AI Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = None

if GEMINI_API_KEY and GEMINI_API_KEY != "AI_KEY_NOT_SET":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini AI configured successfully")
    except Exception as e:
        logger.error(f"Gemini configuration error: {e}")

# Simple text storage (instead of ChromaDB)
documents_store = []

class QueryRequest(BaseModel):
    question: str
    language: str = "hindi"

class QueryResponse(BaseModel):
    answer: str
    success: bool

def extract_text_from_pdf(pdf_file):
    """PDF से टेक्स्ट निकालें"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF पढ़ने में error: {str(e)}")

@app.get("/")
async def health_check():
    return {
        "status": "active", 
        "message": "Study AI Backend is running!",
        "version": "1.0",
        "ai_ready": gemini_model is not None,
        "documents_count": len(documents_store)
    }

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """PDF अपलोड करें और text store करें"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="सिर्फ PDF files allowed हैं")
    
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text:
            raise HTTPException(status_code=400, detail="PDF से कोई टेक्स्ट नहीं मिला")
        
        # Simple storage
        doc_id = str(uuid.uuid4())
        documents_store.append({
            "id": doc_id,
            "filename": file.filename,
            "content": text,
            "size": len(text)
        })
        
        return {
            "success": True,
            "message": f"PDF processed successfully! Text extracted: {len(text)} characters",
            "filename": file.filename,
            "doc_id": doc_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """सवाल पूछें और जवाब पाएं"""
    try:
        if not gemini_model:
            return QueryResponse(
                answer="AI service is currently unavailable. Please configure GEMINI_API_KEY in Render dashboard.",
                success=False
            )
        
        if not documents_store:
            return QueryResponse(
                answer="कृपया पहले PDF documents upload करें।",
                success=False
            )
        
        # Simple context from all documents
        context = "\n\n".join([doc["content"][:1000] for doc in documents_store])  # Limit context size
        
        prompt = f"""
        STUDY MATERIALS CONTEXT:
        {context}

        USER QUESTION: {request.question}

        Instructions:
        1. Answer based on the study materials context
        2. If information is not in context, say "यह जानकारी उपलब्ध नहीं है"
        3. Use simple {request.language} language
        4. Be helpful and educational

        ANSWER:
        """
        
        response = gemini_model.generate_content(prompt)
        
        return QueryResponse(
            answer=response.text,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        return QueryResponse(
            answer=f"Error: {str(e)}",
            success=False
        )

@app.get("/documents")
async def list_documents():
    """Uploaded documents की list दें"""
    return {
        "count": len(documents_store),
        "documents": [
            {"filename": doc["filename"], "size": doc["size"]} 
            for doc in documents_store
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
