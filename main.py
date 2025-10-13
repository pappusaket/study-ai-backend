import os
import uuid
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
import io
import logging
from typing import List, Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Study AI Backend", version="1.0")

# CORS setup - WordPress site के लिए
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production में अपनी website URL डालें
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="study_documents",
        metadata={"description": "Study AI PDF documents collection"}
    )
    logger.info("AI models and database initialized successfully")
except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise e

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")
    
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Gemini configuration error: {e}")
    gemini_model = None

# Request/Response Models
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

class AnswerSubmitRequest(BaseModel):
    user_id: str
    question_id: str
    user_answer: str

class AnswerSubmitResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str
    user_level: str
    performance_score: float
    message: str

# User progress tracking (in production, use database)
user_progress = {}

def extract_text_from_pdf(pdf_file):
    """PDF से टेक्स्ट निकालें"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"Page {page_num + 1}: {page_text}\n\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=400, detail=f"PDF पढ़ने में error: {str(e)}")

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """टेक्स्ट को छोटे chunks में तोड़ें"""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
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
        "correct_answers": 0,
        "total_questions": 0,
        "streak_days": 0,
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
            "streak_days": 0,
            "performance_score": 0.0
        }
    
    user = user_progress[user_id]
    user["total_questions"] += 1
    
    if is_correct:
        user["correct_answers"] += 1
        user["streak_days"] += 1
    else:
        user["streak_days"] = max(0, user["streak_days"] - 1)
    
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "active", 
        "message": "Study AI Backend is running!",
        "version": "1.0"
    }

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """PDF अपलोड करें और process करें"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="सिर्फ PDF files allowed हैं")
    
    try:
        # PDF पढ़ें
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF से कोई टेक्स्ट नहीं मिला")
        
        # Chunks बनाएं
        chunks = split_text_into_chunks(text)
        
        # Embeddings बनाएं और ChromaDB में store करें
        embeddings = embedding_model.encode(chunks).tolist()
        
        # Unique IDs और metadata
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = [{
            "filename": file.filename,
            "chunk_index": i,
            "total_chunks": len(chunks)
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
    """सवाल पूछें और जवाब पाएं"""
    try:
        # सवाल का embedding बनाएं
        question_embedding = embedding_model.encode([request.question]).tolist()[0]
        
        # Similar chunks खोजें
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        if not results['documents'] or not results['documents'][0]:
            return QueryResponse(
                answer="मुझे आपके सवाल का जवाब देने के लिए पर्याप्त जानकारी नहीं मिली। कृपया सुनिश्चित करें कि relevant PDFs upload किए गए हैं।",
                sources=[],
                success=False
            )
        
        # Context बनाएं
        context = "\n\n".join(results['documents'][0])
        sources = [f"Chunk {i+1}" for i in range(len(results['documents'][0]))]
        
        # Gemini को कॉल करें
        if not gemini_model:
            return QueryResponse(
                answer="AI service is currently unavailable. Please try again later.",
                sources=sources,
                success=False
            )
        
        # Language-based prompt
        if request.language == "hindi":
            prompt = f"""
            CONTEXT FROM STUDY MATERIALS:
            {context}
            
            USER QUESTION: {request.question}
            
            निर्देश:
            1. केवल दिए गए CONTEXT से ही जवाब दें
            2. अगर जवाब context में नहीं है, तो कहें "यह जानकारी मेरे पास उपलब्ध नहीं है"
            3. जवाब सरल और स्पष्ट हिंदी में दें
            4. बुलेट पॉइंट्स का उपयोग करें
            5. उदाहरण दें अगर context में available हो
            
            उत्तर:
            """
        else:
            prompt = f"""
            CONTEXT FROM STUDY MATERIALS:
            {context}
            
            USER QUESTION: {request.question}
            
            Instructions:
            1. Answer STRICTLY from the provided CONTEXT only
            2. If answer is not in context, say "This information is not available in my knowledge base"
            3. Keep answer simple and clear
            4. Use bullet points for better readability
            5. Add examples if available in context
            
            ANSWER:
            """
        
        response = gemini_model.generate_content(prompt)
        
        return QueryResponse(
            answer=response.text,
            sources=sources,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return QueryResponse(
            answer=f"Error processing your question: {str(e)}",
            sources=[],
            success=False
        )

@app.post("/adaptive-quiz", response_model=QuizResponse)
async def generate_adaptive_quiz(request: QuizRequest):
    """User के performance के according quiz generate करें"""
    try:
        user_level = get_user_level(request.user_id)
        
        # Level-based content search
        level_keywords = {
            "beginner": ["परिभाषा", "मूल", "आधारभूत", "सरल", "basic", "definition"],
            "intermediate": ["अनुप्रयोग", "तुलना", "प्रक्रिया", "application", "comparison"],
            "advanced": ["विश्लेषण", "महत्व", "जटिल", "analysis", "importance", "complex"]
        }
        
        # Random content लें
        all_docs = collection.get()
        if not all_docs['documents']:
            raise HTTPException(status_code=404, detail="No study materials available")
        
        import random
        random_chunk = random.choice(all_docs['documents'])
        
        # Gemini से quiz generate करवाएं
        quiz_prompt = f"""
        CONTENT: {random_chunk}
        
        इस content से एक multiple choice question (MCQ) generate करें।
        
        REQUIREMENTS:
        - Question language: Hindi
        - 4 options (A, B, C, D)
        - 1 सही जवाब (context के according)
        - 3 गलत जवाब (confusing लेकिन wrong)
        - Difficulty level: {user_level}
        
        FORMAT:
        QUESTION: [question in Hindi]
        OPTIONS:
        A) [option1]
        B) [option2] 
        C) [option3]
        D) [option4]
        CORRECT_ANSWER: [A/B/C/D]
        EXPLANATION: [brief explanation in Hindi]
        
        Generate quiz:
        """
        
        response = gemini_model.generate_content(quiz_prompt)
        response_text = response.text
        
        # Parse response
        lines = response_text.split('\n')
        question = ""
        options = []
        correct_answer = ""
        explanation = ""
        
        current_section = ""
        for line in lines:
            line = line.strip()
            if line.startswith('QUESTION:'):
                current_section = "question"
                question = line.replace('QUESTION:', '').strip()
            elif line.startswith('A)') or line.startswith('B)') or line.startswith('C)') or line.startswith('D)'):
                current_section = "options"
                option_letter = line[0]
                option_text = line[3:].strip()
                options.append({"letter": option_letter, "text": option_text, "value": option_letter})
            elif line.startswith('CORRECT_ANSWER:'):
                correct_answer = line.replace('CORRECT_ANSWER:', '').strip()
            elif line.startswith('EXPLANATION:'):
                current_section = "explanation"
                explanation = line.replace('EXPLANATION:', '').strip()
            elif current_section == "explanation" and line:
                explanation += " " + line
        
        # Validate parsed data
        if not question or len(options) != 4 or not correct_answer:
            raise HTTPException(status_code=500, detail="Failed to generate valid quiz")
        
        question_id = str(uuid.uuid4())
        
        return QuizResponse(
            question=question,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            user_level=user_level,
            question_id=question_id
        )
        
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {str(e)}")

@app.post("/submit-answer", response_model=AnswerSubmitResponse)
async def submit_quiz_answer(request: AnswerSubmitRequest):
    """User का answer check करें और progress update करें"""
    try:
        # Simple answer checking (production में AI-based checking use करें)
        # Note: In production, you'd retrieve the correct answer from storage
        # For now, we'll use a simple approach
        
        # This is a simplified version - in production, you'd store questions and answers
        is_correct = request.user_answer.upper() in ['A', 'B', 'C', 'D']  # Simple validation
        
        # Update user progress
        new_level = update_user_progress(request.user_id, is_correct)
        user = user_progress[request.user_id]
        
        # Generate feedback message
        if is_correct:
            feedback_messages = {
                "beginner": "बहुत अच्छा! आप सही रास्ते पर हैं! 🎉",
                "intermediate": "शानदार! आप concepts अच्छे से समझ रहे हैं! 💪", 
                "advanced": "अद्भुत! आप expert level पर पहुँच रहे हैं! 🚀"
            }
            message = feedback_messages.get(new_level, "Great job! 👍")
        else:
            message = "कोई बात नहीं! कल फिर try करेंगे! Practice makes perfect! 💫"
        
        return AnswerSubmitResponse(
            is_correct=is_correct,
            correct_answer="A",  # In production, get from storage
            explanation="यह सही जवाब है क्योंकि यह study materials के according है।",
            user_level=new_level,
            performance_score=user["performance_score"],
            message=message
        )
        
    except Exception as e:
        logger.error(f"Answer submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Answer processing failed: {str(e)}")

@app.get("/user-progress/{user_id}")
async def get_user_progress(user_id: str):
    """User का progress data get करें"""
    user = user_progress.get(user_id, {
        "current_level": "beginner",
        "correct_answers": 0,
        "total_questions": 0,
        "streak_days": 0,
        "performance_score": 0.0
    })
    
    return {
        "user_id": user_id,
        "current_level": user["current_level"],
        "correct_answers": user["correct_answers"],
        "total_questions": user["total_questions"],
        "streak_days": user["streak_days"],
        "performance_score": user["performance_score"],
        "accuracy_percentage": round(user["performance_score"] * 100, 2)
    }

@app.get("/documents/count")
async def get_documents_count():
    """Available documents की count get करें"""
    try:
        all_docs = collection.get()
        return {
            "total_documents": len(all_docs['documents']),
            "total_chunks": len(all_docs['documents']) if all_docs['documents'] else 0
        }
    except Exception as e:
        logger.error(f"Documents count error: {e}")
        return {"total_documents": 0, "total_chunks": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
