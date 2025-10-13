from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

class QueryResponse(BaseModel):
    answer: str
    success: bool

@app.get("/")
async def root():
    return {"message": "Study AI Backend is running!"}

@app.post("/query")
async def query(request: QueryRequest):
    return {
        "answer": f"Backend working! Your question: {request.question}",
        "success": True
    }

@app.post("/ingest")
async def ingest():
    return {"message": "PDF upload endpoint ready", "success": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
