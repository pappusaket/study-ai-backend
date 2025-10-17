from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise Exception("Missing required environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# ✅ SERVERLESS Pinecone Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "study-ai-index"

# Check if index exists, if not create it
existing_indexes = [index["name"] for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region as per your Pinecone
    )

index = pc.Index(index_name)

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

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    
    return chunks

def create_smart_prompt(question, context, question_lang, content_lang, subject, answer_lang):
    """Create subject and language specific prompt"""
    
    subject_prompts = {
        "maths": """
        QUESTION: {question}
        CONTEXT: {context}
        
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
        CONTEXT: {context}
        
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
        CONTEXT: {context}
        
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
        CONTEXT: {context}
        
        CHEMISTRY ANSWER FORMAT:
        • Chemical reactions और equations दें
        • Elements और compounds के properties
        • Practical applications बताएं
        • Answer in: {answer_lang}
        
        ANSWER:
        """,
        
        "general": """
        QUESTION: {question}
        CONTEXT: {context}
        
        Provide a clear, comprehensive answer in {answer_lang}:
        
        ANSWER:
        """
    }
    
    prompt_template = subject_prompts.get(subject, subject_prompts["general"])
    
    # Add language handling note if needed
    if question_lang != content_lang:
        prompt_template += f"\n\nNOTE: Question is in {question_lang} but content is in {content_lang}. Provide answer in {answer_lang}."
    
    return prompt_template.format(
        question=question,
        context=context,
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
            datetime.now().isoformat(), 'queued', file_size, subject_category
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

# PDF Processing Functions
async def process_pdf_batch(pdf_batch):
    """Process a batch of PDFs"""
    for pdf_data in pdf_batch:
        try:
            # Update status to processing
            update_pdf_status(pdf_data["pdf_id"], "processing")
            
            text = extract_text_from_pdf(pdf_data["file_content"])
            chunks = split_text_into_chunks(text)
            
            for i, chunk in enumerate(chunks):
                # Create embedding
                embedding = genai.embed_content(
                    model="models/embedding-001",
                    content=chunk
                )['embedding']
                
                # Store in Pinecone (Serverless compatible)
                index.upsert(vectors=[(
                    f"{pdf_data['user_id']}_{pdf_data['pdf_id']}_{i}",
                    embedding,
                    {
                        "text": chunk,
                        "filename": pdf_data["filename"],
                        "user_id": pdf_data["user_id"],
                        "pdf_id": pdf_data["pdf_id"],
                        "chunk_index": i,
                        "processed_at": datetime.now().isoformat()
                    }
                )])
            
            # Mark as processed
            update_pdf_status(pdf_data["pdf_id"], "processed", len(chunks))
            
        except Exception as e:
            print(f"Error processing {pdf_data['filename']}: {str(e)}")
            update_pdf_status(pdf_data["pdf_id"], "failed")

async def nightly_processor():
    """Nightly PDF processing from 10 PM to 6 AM"""
    while True:
        current_time = datetime.now().time()
        start_time = datetime.strptime("22:00", "%H:%M").time()
        end_time = datetime.strptime("06:00", "%H:%M").time()
        
        if current_time >= start_time or current_time <= end_time:
            # Process pending PDFs
            all_pending = []
            for user_id, pdfs in pending_pdfs_queue.items():
                all_pending.extend(pdfs)
            
            if all_pending:
                batches = [all_pending[i:i+5] for i in range(0, len(all_pending), 5)]
                for batch in batches:
                    await process_pdf_batch(batch)
                    await asyncio.sleep(60)  # Rate limiting
            
            # Clear processed PDFs from queue
            pending_pdfs_queue.clear()
        
        await asyncio.sleep(300)  # Check every 5 minutes

# API Routes
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    description: str = Form(""),
    subject_category: str = Form("general")
):
    """Upload PDF to processing queue with description"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_content = await file.read()
    file_size = len(file_content)
    
    # Check duplicate
    file_hash = hashlib.md5(file_content).hexdigest()
    
    # Add to database
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
    
    # Add to processing queue
    if user_id not in pending_pdfs_queue:
        pending_pdfs_queue[user_id] = []
    
    pending_pdfs_queue[user_id].append({
        "pdf_id": pdf_id,
        "filename": file.filename,
        "file_content": file_content,
        "user_id": user_id,
        "uploaded_at": datetime.now().isoformat()
    })
    
    return {
        "status": "queued",
        "message": "PDF added to nightly processing queue",
        "expected_ready": "Tomorrow 6 AM",
        "pdf_id": pdf_id
    }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(user_id: str):
    """Get all PDFs for a user"""
    try:
        pdfs = get_user_pdfs(user_id)
        stats = get_pdf_stats(user_id)
        
        return {
            "pdfs": pdfs,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching PDFs: {str(e)}")

@app.delete("/delete-pdf/{pdf_id}")
async def delete_pdf(pdf_id: int, user_id: str):
    """Delete a PDF and its vectors"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get PDF info
        cursor.execute('SELECT filename, user_id FROM pdf_files WHERE id = ? AND user_id = ?', (pdf_id, user_id))
        pdf_info = cursor.fetchone()
        
        if not pdf_info:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        # Delete from Pinecone (Serverless compatible)
        vectors_to_delete = []
        for i in range(1000):  # Assuming max 1000 chunks per PDF
            vector_id = f"{user_id}_{pdf_id}_{i}"
            vectors_to_delete.append(vector_id)
        
        if vectors_to_delete:
            index.delete(ids=vectors_to_delete)
        
        # Delete from database
        cursor.execute('DELETE FROM pdf_files WHERE id = ?', (pdf_id,))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "PDF deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """Ask questions and get instant answers"""
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
        
        # Create question embedding
        question_embedding = genai.embed_content(
            model="models/embedding-001",
            content=request.question
        )['embedding']
        
        # Search in Pinecone (Serverless compatible)
        results = index.query(
            vector=question_embedding,
            top_k=5,
            include_metadata=True,
            filter={"user_id": request.user_id}
        )
        
        if not results['matches']:
            return {
                "answer": "I couldn't find relevant information in your PDFs to answer this question.",
                "processing_time": f"{time.time() - start_time:.2f}s",
                "subject": subject,
                "language": answer_lang,
                "sources_used": 0
            }
        
        # Combine context
        context_chunks = [match['metadata']['text'] for match in results['matches']]
        context = "\n\n".join(context_chunks)
        content_lang = detect_language(context)
        
        # Generate answer
        prompt = create_smart_prompt(
            question=request.question,
            context=context,
            question_lang=question_lang,
            content_lang=content_lang,
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
            "sources_used": len(results['matches'])
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

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "active", 
        "service": "Study AI Backend",
        "timestamp": datetime.now().isoformat()
    }

# Start background tasks
@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    asyncio.create_task(nightly_processor())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
