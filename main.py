from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI(title="Study AI - Python 3.11 Compatible")

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
    return {"status": "active", "message": "Study AI - Python 3.11 Compatible"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "python_version": "3.11.9",
        "gemini": "configured" if GEMINI_API_KEY else "not_configured"
    }

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
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
