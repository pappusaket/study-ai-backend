from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

# ✅ PEHLE APP CREATE KARO
app = FastAPI(title="Study AI - Basic")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ AB @app.get USE KARO
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
            }
        </style>
    </head>
    <body>
        <div class="status-box">
            <h1>🚀 Study AI Backend</h1>
            <div class="status">
                <span class="live-dot"></span> LIVE & RUNNING
            </div>
            <p>Server is active and responding to requests</p>
            
            <div class="endpoints">
                <a href="/health">🔍 Health Check</a>
                <a href="/docs">📚 API Documentation</a>
                <a href="/test">🧪 Test Endpoint</a>
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

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

@app.get("/health")
async def health():
    gemini_status = "configured" if GEMINI_API_KEY else "not_configured"
    return {
        "status": "healthy", 
        "service": "Study AI",
        "gemini": gemini_status
    }

@app.get("/test")
async def test():
    return {"test": "passed", "message": "Everything is working!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
