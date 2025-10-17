from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI(title="Study AI - Basic")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"

@app.get("/")
async def root():
    return {"message": "Study AI Backend is working!", "status": "success"}

@app.get("/health")
async def health():
    gemini_status = "configured" if GEMINI_API_KEY else "not_configured"
    return {
        "status": "healthy", 
        "service": "Study AI",
        "gemini": gemini_status
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Answer this question: {request.question}")
        
        return {
            "question": request.question,
            "answer": response.text,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/test")
async def test():
    return {"test": "passed", "message": "Everything is working!"}
