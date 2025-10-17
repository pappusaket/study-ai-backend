from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import hashlib
import io
from pypdf import PdfReader
from datetime import datetime, timedelta
import asyncio
import time
import os
from typing import List, Optional, Dict
import schedule
import json
import sqlite3
from pathlib import Path

# Initialize FastAPI
app = FastAPI(title="Study AI Backend")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FREE APIs Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise Exception("Missing GEMINI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)

# ✅ TEMPORARY: Pinecone disabled for now
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# if not PINECONE_API_KEY:
#     raise Exception("Missing PINECONE_API_KEY environment variable")

print("✅ Gemini AI configured successfully")
print("ℹ️  Pinecone vector search temporarily disabled")

# SQLite Database Setup
DB_PATH = "study_ai.db"

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # PDFs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            description TEXT,
            upload_date TEXT NOT NULL,
            status TEXT NOT NULL,
            processed_date TEXT,
            chunks_count INTEGER DEFAULT 0,
            file_size INTEGER,
            subject_category TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

# Initialize database
init_db()

# Global storage
user_progress = {}
pending_pdfs_queue = {}
user_language_prefs = {}

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"
    language_preference: str = "auto"

class QuizRequest(BaseModel):
    user_id: str
    subject: str = "auto"
    difficulty: str = "auto"

class UploadRequest(BaseModel):
    user_id: str
    description: Optional[str] = ""
    subject_category: Optional[str] = "general"

class PDFInfo(BaseModel):
    id: int
    filename: str
    description: str
    upload_date: str
    status: str
    processed_date: Optional[str]
    chunks_count: int
    file_size: int
    subject_category: str

# Utility Functions
def detect_language(text):
    """Detect language of text"""
    hindi_chars = set('अआइईउऊऋएऐओऔकखगघचछजझटठडढतथधनपफबभमयरलवशषसह')
    has_hindi = any(char in hindi_chars for char in text)
    has_english = any(char.isalpha() for char in text)
    
    if has_hindi and not has_english:
        return "hindi"
    elif has_english and not has_hindi:
        return "english"
    else:
        return "mixed"

def detect_subject(question, content=""):
    """Detect subject from question and content"""
    subject_keywords = {
        "maths": ["गणित", "गणना", "समीकरण", "संख्या", "बीजगणित", "ज्यामिति", "math", "calculate", "equation"],
        "physics": ["भौतिकी", "बल", "गति", "ऊर्जा", "विद्युत", "चुंबक", "प्रकाश", "physics", "force", "energy"],
        "chemistry": ["रसायन", "तत्व", "यौगिक", "अभिक्रिया", "आवर्त", "chemistry", "element", "compound"],
        "biology": ["जीवविज्ञान", "जीव", "कोशिका", "डीएनए", "परिस्थितिकी", "biology", "cell", "dna"],
        "history": ["इतिहास", "युद्ध", "सम्राट", "सभ्यता", "काल", "history", "war", "empire"],
        "geography": ["भूगोल", "मानचित्र", "जलवायु", "पर्वत", "नदी", "geography", "map", "climate"]
    }
    
    combined_text = question.lower() + " " + content.lower()
    
    for subject, keywords in subject_keywords.items():
        if any(keyword in combined_text for keyword in keywords):
            return subject
    
    return "general"

def get_answer_language(preference, question_lang, user_id):
    """Determine answer language"""
    if preference != "auto":
        return preference
    
    # Store user preference
    user_language_prefs[user_id] = question_lang
    return question_lang

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF"""
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF reading error: {str(e)}")

def create_smart_prompt(question, question_lang, subject, answer_lang):
    """Create subject and language specific prompt"""
    
    subject_prompts = {
        "maths": """
        QUESTION: {question}
        
        MATHS ANSWER FORMAT:
        • Direct और concise answer दें
        • Formulas और calculations clear दें  
        • Step-by-step solution provide करें
        • Examples with numbers दें
        • Answer in: {answer_lang}
        
        ANSWER:
        """,
        
        "physics": """
        QUESTION: {question}
        
        PHYSICS ANSWER FORMAT:
        • Conceptual explanation दें
        • Laws और principles mention करें
        • Real-world examples दें
        • Formulas with units दें
        • Answer in: {answer_lang}
        
        ANSWER:
        """,
        
        "biology": """
        QUESTION: {question}
        
        BIOLOGY ANSWER FORMAT:
        • Detailed processes explain करें
        • Step-by-step explanations दें
        • Structure descriptions दें
        • Importance और applications बताएं
        • Answer in: {answer_lang}
        
        ANSWER:
        """,
        
        "chemistry": """
        QUESTION: {question}
        
        CHEMISTRY ANSWER FORMAT:
        • Chemical reactions और equations दें
        • Elements और compounds के properties
        • Practical applications बताएं
        • Answer in: {answer_lang}
        
        ANSWER:
        """,
        
        "general": """
        QUESTION: {question}
        
        Provide a clear, comprehensive answer in {answer_lang}:
        - Explain concepts simply
        - Provide examples if relevant
        - Make it educational and easy to understand
        
        ANSWER:
        """
    }
    
    prompt_template = subject_prompts.get(subject, subject_prompts["general"])
    
    return prompt_template.format(
        question=question,
        answer_lang=answer_lang
    )

# Database Functions
def add_pdf_to_db(filename, file_hash, user_id, description, file_size, subject_category="general"):
    """Add PDF info to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO pdf_files 
            (filename, file_hash, user_id, description, upload_date, status, file_size, subject_category)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename, file_hash, user_id, description, 
            datetime.now().isoformat(), 'stored', file_size, subject_category
        ))
        
        conn.commit()
        pdf_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # Duplicate file
        cursor.execute('SELECT id FROM pdf_files WHERE file_hash = ?', (file_hash,))
        result = cursor.fetchone()
        pdf_id = result[0] if result else None
    finally:
        conn.close()
    
    return pdf_id

def update_pdf_status(pdf_id, status, chunks_count=0):
    """Update PDF processing status"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if status == 'processed':
        cursor.execute('''
            UPDATE pdf_files 
            SET status = ?, processed_date = ?, chunks_count = ?
            WHERE id = ?
        ''', (status, datetime.now().isoformat(), chunks_count, pdf_id))
    else:
        cursor.execute('''
            UPDATE pdf_files 
            SET status = ?
            WHERE id = ?
        ''', (status, pdf_id))
    
    conn.commit()
    conn.close()

def get_user_pdfs(user_id):
    """Get all PDFs for a user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, description, upload_date, status, processed_date, chunks_count, file_size, subject_category
        FROM pdf_files 
        WHERE user_id = ? 
        ORDER BY upload_date DESC
    ''', (user_id,))
    
    pdfs = []
    for row in cursor.fetchall():
        pdfs.append({
            "id": row[0],
            "filename": row[1],
            "description": row[2],
            "upload_date": row[3],
            "status": row[4],
            "processed_date": row[5],
            "chunks_count": row[6],
            "file_size": row[7],
            "subject_category": row[8]
        })
    
    conn.close()
    return pdfs

def get_pdf_stats(user_id):
    """Get PDF statistics for user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed,
            SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
        FROM pdf_files 
        WHERE user_id = ?
    ''', (user_id,))
    
    result = cursor.fetchone()
    stats = {
        "total": result[0],
        "processed": result[1],
        "queued": result[2],
        "processing": result[3],
        "failed": result[4]
    }
    
    conn.close()
    return stats

# API Routes
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    description: str = Form(""),
    subject_category: str = Form("general")
):
    """Upload PDF - store info but don't process temporarily"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_content = await file.read()
    file_size = len(file_content)
    
    # Check duplicate
    file_hash = hashlib.md5(file_content).hexdigest()
    
    # Add to database only
    pdf_id = add_pdf_to_db(
        filename=file.filename,
        file_hash=file_hash,
        user_id=user_id,
        description=description,
        file_size=file_size,
        subject_category=subject_category
    )
    
    if pdf_id is None:
        return {
            "status": "duplicate", 
            "message": "This PDF has already been uploaded",
            "pdf_id": None
        }
    
    return {
        "status": "success",
        "message": "PDF stored successfully. Vector search will be available soon.",
        "pdf_id": pdf_id,
        "note": "PDF search temporarily disabled - coming soon"
    }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(user_id: str):
    """Get all PDFs for a user"""
    try:
        pdfs = get_user_pdfs(user_id)
        stats = get_pdf_stats(user_id)
        
        return {
            "pdfs": pdfs,
            "stats": stats,
            "note": "PDF search temporarily disabled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching PDFs: {str(e)}")

@app.delete("/delete-pdf/{pdf_id}")
async def delete_pdf(pdf_id: int, user_id: str):
    """Delete a PDF from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get PDF info
        cursor.execute('SELECT filename, user_id FROM pdf_files WHERE id = ? AND user_id = ?', (pdf_id, user_id))
        pdf_info = cursor.fetchone()
        
        if not pdf_info:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Delete from database only (Pinecone temporarily disabled)
        cursor.execute('DELETE FROM pdf_files WHERE id = ?', (pdf_id,))
        conn.commit()
        conn.close()
        
        return {
            "status": "success", 
            "message": "PDF deleted successfully",
            "note": "Vector data cleanup will happen when search is enabled"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """Ask questions and get instant answers (without PDF search temporarily)"""
    start_time = time.time()
    
    try:
        # Detect languages
        question_lang = detect_language(request.question)
        answer_lang = get_answer_language(
            request.language_preference, 
            question_lang, 
            request.user_id
        )
        
        # Detect subject
        subject = detect_subject(request.question)
        
        # Direct Gemini se answer lo (without PDF context temporarily)
        prompt = create_smart_prompt(
            question=request.question,
            question_lang=question_lang,
            subject=subject,
            answer_lang=answer_lang
        )
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Update user progress
        if request.user_id not in user_progress:
            user_progress[request.user_id] = {
                "questions_asked": 0,
                "subjects": {},
                "last_active": datetime.now().isoformat()
            }
        
        user_progress[request.user_id]["questions_asked"] += 1
        user_progress[request.user_id]["subjects"][subject] = \
            user_progress[request.user_id]["subjects"].get(subject, 0) + 1
        
        return {
            "answer": response.text,
            "processing_time": f"{time.time() - start_time:.2f}s",
            "subject": subject,
            "language": answer_lang,
            "sources_used": 0,
            "note": "PDF search temporarily disabled - using general knowledge"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/daily-quiz")
async def generate_daily_quiz(request: QuizRequest):
    """Generate adaptive daily quiz"""
    try:
        # Get user level
        user_level = "beginner"
        if request.user_id in user_progress:
            total_questions = user_progress[request.user_id]["questions_asked"]
            if total_questions > 50:
                user_level = "advanced"
            elif total_questions > 20:
                user_level = "intermediate"
        
        # Determine subject and difficulty
        subject = request.subject
        difficulty = request.difficulty
        
        if subject == "auto":
            # Find user's weak subject
            if request.user_id in user_progress:
                subjects = user_progress[request.user_id]["subjects"]
                subject = min(subjects, key=subjects.get) if subjects else "general"
            else:
                subject = "general"
        
        if difficulty == "auto":
            difficulty = user_level
        
        # Get user language preference
        user_lang = user_language_prefs.get(request.user_id, "hindi")
        
        # Generate quiz
        quiz_prompt = f"""
        Generate a {difficulty} level {subject} multiple choice question with:
        - 1 correct answer
        - 3 wrong but confusing options
        - All options in {user_lang} language
        - Educational and interesting content
        
        FORMAT:
        QUESTION: [question text]
        OPTION_A: [wrong but confusing option]
        OPTION_B: [correct answer] 
        OPTION_C: [wrong but relevant option]
        OPTION_D: [wrong but similar option]
        ANSWER: B
        EXPLANATION: [brief explanation in {user_lang}]
        """
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(quiz_prompt)
        
        # Parse response
        lines = response.text.split('\n')
        quiz_data = {}
        current_key = ""
        
        for line in lines:
            if line.startswith('QUESTION:'):
                quiz_data['question'] = line.replace('QUESTION:', '').strip()
                current_key = 'question'
            elif line.startswith('OPTION_A:'):
                quiz_data['options'] = [line.replace('OPTION_A:', '').strip()]
                current_key = 'options'
            elif line.startswith('OPTION_B:'):
                quiz_data['options'].append(line.replace('OPTION_B:', '').strip())
            elif line.startswith('OPTION_C:'):
                quiz_data['options'].append(line.replace('OPTION_C:', '').strip())
            elif line.startswith('OPTION_D:'):
                quiz_data['options'].append(line.replace('OPTION_D:', '').strip())
            elif line.startswith('ANSWER:'):
                quiz_data['answer'] = line.replace('ANSWER:', '').strip()
            elif line.startswith('EXPLANATION:'):
                quiz_data['explanation'] = line.replace('EXPLANATION:', '').strip()
            elif current_key == 'options' and line.strip():
                quiz_data['options'][-1] += " " + line.strip()
        
        return {
            "question": quiz_data.get("question", "Quiz question"),
            "options": quiz_data.get("options", ["Option A", "Option B", "Option C", "Option D"]),
            "correct_answer": quiz_data.get("answer", "B"),
            "explanation": quiz_data.get("explanation", "Explanation"),
            "difficulty": difficulty,
            "subject": subject,
            "language": user_lang
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

@app.get("/user-progress/{user_id}")
async def get_user_progress(user_id: str):
    """Get user learning progress"""
    return user_progress.get(user_id, {
        "questions_asked": 0,
        "subjects": {},
        "last_active": datetime.now().isoformat()
    })

@app.get("/health")
async def detailed_health_check():
    """Detailed health check for monitoring"""
    try:
        # Check database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        db_status = "healthy"
        conn.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    try:
        # Check Gemini
        genai.list_models()
        gemini_status = "healthy"
    except Exception as e:
        gemini_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "active",
        "service": "Study AI Backend",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "gemini_ai": gemini_status,
        "pinecone": "temporarily_disabled",
        "active_users": len(user_progress),
        "version": "1.0",
        "note": "PDF vector search coming soon"
    }

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "active", 
        "service": "Study AI Backend",
        "timestamp": datetime.now().isoformat(),
        "message": "Server is running successfully!",
        "note": "PDF search temporarily disabled"
    }

# Startup event - background tasks temporarily disabled
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    try:
        print("🚀 Starting Study AI Backend...")
        print("✅ Gemini AI configured")
        print("ℹ️  Pinecone vector search temporarily disabled")
        print("✅ Database initialized")
        print("✅ All routes loaded successfully")
        print("🔧 Server ready to handle requests")
        
        # Background tasks temporarily disabled
        # asyncio.create_task(nightly_processor())
        
    except Exception as e:
        print(f"💥 Startup error: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
