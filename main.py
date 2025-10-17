from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Study AI - Basic")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Study AI Backend is working!", "status": "success"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Study AI"}

@app.get("/test")
async def test():
    return {"test": "passed", "message": "Everything is working!"}
