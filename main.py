import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

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

class QueryRequest(BaseModel):
    question: str
    language: str = "hindi"

class QueryResponse(BaseModel):
    answer: str
    success: bool

@app.get("/")
async def health_check():
    return {
        "status": "active", 
        "message": "Study AI Backend is running!",
        "version": "1.0",
        "ai_ready": False
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Basic response - AI will be added later
        answer = f"🚀 Study AI System is ready! Your question: '{request.question}'. AI features will be enabled in the next update."
        
        return QueryResponse(
            answer=answer,
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
        "message": "PDF upload system will be available in next update.",
        "status": "backend_ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
