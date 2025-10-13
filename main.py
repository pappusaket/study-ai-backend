import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
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

class QueryRequest(BaseModel):
    question: str
    language: str = "hindi"

class QueryResponse(BaseModel):
    answer: str
    success: bool

@app.get("/")
async def health_check():
    return {"status": "active", "message": "Study AI Backend is running!"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        if not gemini_model:
            return QueryResponse(
                answer="AI service is currently unavailable. Please configure GEMINI_API_KEY.",
                success=False
            )
        
        prompt = f"""
        USER QUESTION: {request.question}
        
        You are a helpful AI assistant. Answer the question in {request.language}.
        Keep the answer simple and clear.
        
        ANSWER:
        """
        
        response = gemini_model.generate_content(prompt)
        
        return QueryResponse(
            answer=response.text,
            success=True
        )
        
    except Exception as e:
        return QueryResponse(
            answer=f"Error: {str(e)}",
            success=False
        )

@app.post("/ingest")
async def ingest_pdf():
    return {
        "success": True,
        "message": "PDF upload will be available in next update. Basic AI chat is working!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
