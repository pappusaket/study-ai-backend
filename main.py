from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import os

# ✅ PEHLE APP CREATE KARO
app = FastAPI(title="Study AI - With Q&A")

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

# Pydantic Model for Q&A
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Study AI - Status</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                padding: 50px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .status-box { 
                background: rgba(255,255,255,0.1); 
                padding: 30px; 
                border-radius: 15px; 
                display: inline-block;
                backdrop-filter: blur(10px);
                max-width: 500px;
            }
            .status { 
                font-size: 24px; 
                margin: 20px 0; 
            }
            .live-dot {
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #00ff00;
                border-radius: 50%;
                animation: pulse 2s infinite;
                margin-right: 10px;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .endpoints {
                margin-top: 20px;
            }
            a {
                color: #00ff00;
                text-decoration: none;
                display: block;
                margin: 10px;
                padding: 10px;
                background: rgba(0,0,0,0.3);
                border-radius: 5px;
                transition: all 0.3s;
            }
            a:hover {
                background: rgba(0,0,0,0.5);
                transform: translateY(-2px);
            }
            .feature-badge {
                background: #00ff00;
                color: #333;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
        </style>
    </head>
    <body>
        <div class="status-box">
            <h1>🚀 Study AI Backend</h1>
            <div class="status">
                <span class="live-dot"></span> LIVE & RUNNING
            </div>
            <p>Smart Q&A System with AI-Powered Answers</p>
            
            <div class="endpoints">
                <a href="/health">🔍 Health Check</a>
                <a href="/docs">📚 API Documentation <span class="feature-badge">NEW</span></a>
                <a href="/test">🧪 Test Endpoint</a>
            </div>

            <div style="margin-top: 30px; padding: 20px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                <h3>🎯 Available Features</h3>
                <p>✅ Smart Q&A System</p>
                <p>✅ Multi-language Support</p>
                <p>✅ Fast Response Time</p>
                <p style="font-size: 12px; opacity: 0.8;">Use /ask endpoint for questions</p>
            </div>
            
            <div style="margin-top: 20px; font-size: 12px; opacity: 0.8;">
                Last checked: <span id="timestamp">loading...</span>
            </div>
        </div>

        <script>
            function updateTime() {
                document.getElementById('timestamp').textContent = new Date().toLocaleString();
            }
            updateTime();
            setInterval(updateTime, 1000);
        </script>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    gemini_status = "configured" if GEMINI_API_KEY else "not_configured"
    return {
        "status": "healthy", 
        "service": "Study AI",
        "gemini": gemini_status,
        "features": ["Q&A System", "Multi-language", "Fast API"]
    }

@app.get("/test")
async def test():
    return {"test": "passed", "message": "Everything is working!"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask questions and get AI-powered answers"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        # Create prompt with context
        prompt = f"""
        You are a helpful study assistant. Answer the following question clearly and concisely.
        
        QUESTION: {request.question}
        
        Provide a comprehensive answer that would help a student understand the concept.
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return {
            "question": request.question,
            "answer": response.text,
            "status": "success",
            "user_id": request.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# New endpoint for quick testing
@app.get("/ask-simple")
async def ask_simple(question: str):
    """Quick test endpoint for questions"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(question)
        
        return {
            "question": question,
            "answer": response.text,
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
