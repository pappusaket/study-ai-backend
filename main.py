import os
import uuid
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Note: Pydantic v1 is used for compatibility (pydantic==1.10.13)
from pydantic import BaseModel 
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
import io
import logging
from typing import List
import random # For random quiz chunk selection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Study AI Backend", version="1.0")

# CORS setup - Production में अपनी website URL डालें
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production में अपनी website URL डालें
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    # 1. Embedding Model (Local & Free)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. ChromaDB (Local/Persistent File - RAG)
    # Render's free tier has a non-persistent disk. Data will reset on sleep/deploy.
    # For testing, we use in-memory or file-based for local persistence simulation.
    chroma_client = chromadb.PersistentClient(path="./chroma_db") 
    collection = chroma_client.get_or_create_collection(
        name="study_documents",
        metadata={"description": "Study AI PDF documents collection"}
    )
    logger.info("AI models and database initialized successfully")
except Exception as e:
    logger.error(f"Initialization error: {e}")
    # Application will likely fail if initialization fails.
    raise e

# Configure Gemini (Free Tier LLM)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. RAG/Quiz will fail.")
    
gemini_model = None
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini Model configured successfully.")
except Exception as e:
    logger.error(f"Gemini configuration error: {e}")
    gemini_model = None

# --- Pydantic v1 Models ---
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
    topic: str = "random"

class QuizResponse(BaseModel):
    question: str
    options: List[dict]
    correct_answer: str
    explanation: str
    user_level: str
    question_id: str

# User progress tracking (Temporary: Use a proper database like Firestore in production)
user_progress = {}

def extract_text_from_pdf(pdf_file):
    """PDF से टेक्स्ट निकालें"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            # Include page number in the context for better sourcing
            page_text = page.extract_text()
            if page_text:
                text += f"[Page {page_num + 1}]: {page_text}\n\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=400, detail=f"PDF पढ़ने में error: {str(e)}")

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """टेक्स्ट को छोटे chunks में तोड़ें"""
    # Simple word-based splitting logic
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def get_user_level(user_id):
    """User का current level get करें"""
    user = user_progress.get(user_id, {
        "current_level": "beginner",
        "performance_score": 0.0
    })
    return user["current_level"]

def update_user_progress(user_id, is_correct):
    """User का progress update करें"""
    if user_id not in user_progress:
        user_progress[user_id] = {
            "current_level": "beginner",
            "correct_answers": 0,
            "total_questions": 0,
            "performance_score": 0.0
        }
    
    user = user_progress[user_id]
    user["total_questions"] += 1
    
    if is_correct:
        user["correct_answers"] += 1
    
    # Calculate performance score
    if user["total_questions"] > 0:
        user["performance_score"] = user["correct_answers"] / user["total_questions"]
    
    # Update level based on performance
    if user["performance_score"] < 0.5:
        user["current_level"] = "beginner"
    elif user["performance_score"] < 0.75:
        user["current_level"] = "intermediate"
    else:
        user["current_level"] = "advanced"
    
    return user["current_level"]

# --- API Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active", 
        "message": "Study AI Backend is running!",
        "version": "1.0",
        "ai_ready": gemini_model is not None,
        "docs_count": collection.count()
    }

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """PDF अपलोड करें और process करें"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="सिर्फ PDF files allowed हैं")
    
    if collection.count() > 500: # Simple guardrail for free tier
        raise HTTPException(status_code=429, detail="Maximum document chunks reached. Delete some first.")
    
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF से कोई टेक्स्ट नहीं मिला")
        
        chunks = split_text_into_chunks(text)
        
        # Embeddings बनाएं
        embeddings = embedding_model.encode(chunks).tolist()
        
        # Unique IDs और metadata
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = [{
            "filename": file.filename,
            "chunk_index": i
        } for i in range(len(chunks))]
        
        # ChromaDB में add करें
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"PDF processed successfully: {file.filename}, {len(chunks)} chunks")
        
        return {
            "success": True,
            "message": f"PDF successfully processed! {len(chunks)} chunks added.",
            "filename": file.filename,
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """सवाल पूछें और जवाब पाएं (RAG Logic)"""
    if not gemini_model:
        return QueryResponse(answer="AI service (Gemini) is not configured.", sources=[], success=False)

    try:
        # 1. Retrieval
        question_embedding = embedding_model.encode([request.question]).tolist()[0]
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
        
        context = "\n\n".join(results['documents'][0])
        # Simple source identification (can be improved by parsing page numbers from metadata)
        sources = [f"Source {i+1}" for i in range(len(results['documents'][0]))]
        
        # 2. Augmented Generation Prompt (Hindi/English)
        prompt_instructions = f"""
        Instructions:
        1. Answer STRICTLY from the provided CONTEXT only.
        2. If the answer is not in the context, say "यह जानकारी मेरे पास उपलब्ध नहीं है" (or "This information is not available")
        3. Use simple, clear language (Hindi/English based on user's request).
        4. Use bullet points or numbered lists for better readability.
        """

        if request.language.lower() == "hindi":
            system_prompt = f"आप एक सहायक अध्ययन ट्यूटर हैं। उपयोगकर्ता के सवाल का जवाब केवल हिंदी में, दिए गए संदर्भ का उपयोग करके दें।"
            question_text = f"उपयोगकर्ता का सवाल: {request.question}"
        else:
            system_prompt = f"You are a helpful study tutor. Answer the user's question only in English, using the context provided."
            question_text = f"USER QUESTION: {request.question}"

        full_prompt = f"""
        System Instruction: {system_prompt}

        CONTEXT FROM STUDY MATERIALS:
        {context}
        
        {prompt_instructions}
        
        {question_text}
        
        Answer:
        """
        
        # 3. Generation
        response = gemini_model.generate_content(full_prompt)
        
        return QueryResponse(
            answer=response.text,
            sources=sources,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing your question: {str(e)}")

@app.post("/adaptive-quiz", response_model=QuizResponse)
async def generate_adaptive_quiz(request: QuizRequest):
    """User के performance के according quiz generate करें (Hindi Output)"""
    if not gemini_model:
        raise HTTPException(status_code=503, detail="AI service (Gemini) is not configured.")
        
    try:
        user_level = get_user_level(request.user_id)
        
        # Random content लें
        all_docs = collection.get()
        if not all_docs['documents']:
            raise HTTPException(status_code=404, detail="No study materials available")
        
        random_chunk = random.choice(all_docs['documents'])
        
        # Gemini से structured quiz generate करवाएं
        quiz_prompt = f"""
        You are a Quiz Generation AI. Use the provided CONTENT to generate ONE multiple choice question (MCQ).
        
        REQUIREMENTS:
        - Question language: Hindi
        - Difficulty level: {user_level} (make it harder if 'advanced', easier if 'beginner')
        - Output format MUST be a single, raw JSON object.
        
        CONTENT: {random_chunk}
        
        JSON SCHEMA:
        {{
            "question": "[question in Hindi]",
            "options": [
                {{"letter": "A", "text": "[option text]"}},
                {{"letter": "B", "text": "[option text]"}},
                {{"letter": "C", "text": "[option text]"}},
                {{"letter": "D", "text": "[option text]"}}
            ],
            "correct_answer": "[A/B/C/D]",
            "explanation": "[brief explanation in Hindi]"
        }}
        
        Generate the JSON object based on the CONTENT and DIFFICULTY LEVEL:
        """
        
        # We will use text generation and JSON parsing (since the request wasn't for structured API call)
        response = gemini_model.generate_content(quiz_prompt)
        response_text = response.text.strip()
        
        # Attempt to clean and parse JSON (LLMs sometimes wrap JSON in markdown ```json)
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()

        quiz_data = json.loads(response_text)
        
        question_id = str(uuid.uuid4())
        
        return QuizResponse(
            question=quiz_data['question'],
            options=quiz_data['options'],
            correct_answer=quiz_data['correct_answer'],
            explanation=quiz_data['explanation'],
            user_level=user_level,
            question_id=question_id
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Quiz generation error: Failed to parse JSON: {e}")
        raise HTTPException(status_code=500, detail="Quiz generation failed (AI response not valid JSON).")
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")

# Add /submit-answer and /user-progress endpoints from your original plan here...
# For brevity, we assume the core RAG logic is the current focus.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
