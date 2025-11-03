from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import google.generativeai as genai
import hashlib
import io
from pypdf import PdfReader
import os
import sqlite3
from datetime import datetime
from typing import Optional
import re
import secrets
import logging
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Study AI - SpeechSynthesis TTS")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Configuration
class Settings:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.api_secret_key = os.getenv("API_SECRET_KEY", "default-secret-key-change-in-production")
        self.database_url = os.getenv("DATABASE_URL", "study_ai.db")
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

settings = Settings()
genai.configure(api_key=settings.gemini_api_key)

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not secrets.compare_digest(credentials.credentials, settings.api_secret_key):
        logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Custom Exception
class CustomHTTPException(HTTPException):
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

@app.exception_handler(CustomHTTPException)
async def custom_http_exception_handler(request: Request, exc: CustomHTTPException):
    logger.error(f"Error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "error_code": exc.error_code}
    )

# Initialize Database
def init_db():
    """Initialize database"""
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        # PDF files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                description TEXT,
                upload_date TEXT NOT NULL,
                status TEXT NOT NULL,
                file_size INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                text_content TEXT
            )
        ''')
        
        # Text files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wordpress_file_id INTEGER,
                filename TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                description TEXT,
                upload_date TEXT NOT NULL,
                file_size INTEGER,
                processed BOOLEAN DEFAULT TRUE,
                text_content TEXT,
                file_type TEXT
            )
        ''')
        
        # Usage logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Study AI Application started successfully")

# Pydantic Models
class QuestionRequest(BaseModel):
    question: str
    user_id: str = "default"
    use_pdf_context: bool = True
    language: str = "auto"

class TextFileRequest(BaseModel):
    file_id: int
    filename: str
    content: str
    user_id: str
    file_type: str = "text"

class PDFInfo(BaseModel):
    id: int
    filename: str
    description: str
    upload_date: str
    status: str
    file_size: int
    processed: bool

# Utility Functions
def extract_text_from_pdf(pdf_content):
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise CustomHTTPException(status_code=400, detail=f"PDF reading error: {str(e)}")

def validate_pdf_file(file_content: bytes) -> bool:
    """Validate PDF file"""
    try:
        # Basic PDF validation - check if it starts with PDF header
        return file_content.startswith(b'%PDF')
    except Exception as e:
        logger.error(f"File validation error: {e}")
        return False

async def log_usage(user_id: str, endpoint: str, request: Request):
    """Log API usage"""
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO usage_logs (user_id, endpoint, timestamp, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, endpoint, datetime.now().isoformat(), 
              request.client.host if request.client else None,
              request.headers.get("user-agent")))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Usage logging failed: {e}")

def find_relevant_text_simple(question, user_id):
    """Simple keyword-based search for relevant text from both PDFs and text files"""
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT text_content, filename 
            FROM pdf_files 
            WHERE user_id = ? AND processed = TRUE AND text_content IS NOT NULL
        ''', (user_id,))
        
        pdfs = cursor.fetchall()
        
        cursor.execute('''
            SELECT text_content, filename 
            FROM text_files 
            WHERE user_id = ? AND processed = TRUE AND text_content IS NOT NULL
        ''', (user_id,))
        
        text_files = cursor.fetchall()
        
        conn.close()
        
        all_files = pdfs + text_files
        
        if not all_files:
            return []
        
        question_keywords = set(re.findall(r'\w+', question.lower()))
        relevant_sections = []
        
        for text_content, filename in all_files:
            if not text_content:
                continue
                
            sentences = re.split(r'[.!?]+', text_content)
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                    
                sentence_keywords = set(re.findall(r'\w+', sentence.lower()))
                common_keywords = question_keywords.intersection(sentence_keywords)
                
                if len(common_keywords) >= 2:
                    relevance_score = len(common_keywords) / len(question_keywords)
                    relevant_sections.append({
                        'text': sentence.strip(),
                        'filename': filename,
                        'score': relevance_score
                    })
        
        relevant_sections.sort(key=lambda x: x['score'], reverse=True)
        return relevant_sections[:3]
    except Exception as e:
        logger.error(f"Relevant text search error: {e}")
        return []

def process_pdf_content(pdf_id, text_content):
    """Store PDF text content"""
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE pdf_files SET processed = TRUE, text_content = ? WHERE id = ?',
            (text_content, pdf_id)
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error storing PDF content: {e}")
        return False

def add_pdf_to_db(filename, file_hash, user_id, description, file_size):
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        # Check for duplicate
        cursor.execute('SELECT id FROM pdf_files WHERE file_hash = ?', (file_hash,))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return existing[0]  # Return existing ID
        
        # Insert new PDF
        cursor.execute('''
            INSERT INTO pdf_files 
            (filename, file_hash, user_id, description, upload_date, status, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, file_hash, user_id, description, 
              datetime.now().isoformat(), 'uploaded', file_size))
        
        conn.commit()
        pdf_id = cursor.lastrowid
        conn.close()
        
        return pdf_id
    except Exception as e:
        logger.error(f"Error adding PDF to database: {e}")
        raise CustomHTTPException(status_code=500, detail="Database error")

def get_user_pdfs(user_id):
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, filename, description, upload_date, status, file_size, processed
            FROM pdf_files 
            WHERE user_id = ? 
            ORDER BY upload_date DESC
        ''', (user_id,))
        
        pdfs = []
        for row in cursor.fetchall():
            pdfs.append({
                "id": row[0], "filename": row[1], "description": row[2],
                "upload_date": row[3], "status": row[4], "file_size": row[5],
                "processed": bool(row[6])
            })
        conn.close()
        return pdfs
    except Exception as e:
        logger.error(f"Error getting user PDFs: {e}")
        return []

def process_text_file_content(file_data: TextFileRequest):
    """Process and store text file content from WordPress"""
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        file_hash = hashlib.md5(file_data.content.encode()).hexdigest()
        
        # Check for duplicate
        cursor.execute('SELECT id FROM text_files WHERE file_hash = ? AND user_id = ?', 
                      (file_hash, file_data.user_id))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return {"status": "duplicate", "file_id": existing[0]}
        
        # Insert new text file
        cursor.execute('''
            INSERT INTO text_files 
            (wordpress_file_id, filename, file_hash, user_id, description, upload_date, file_size, text_content, file_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_data.file_id,
            file_data.filename,
            file_hash,
            file_data.user_id,
            f"Text file from WordPress: {file_data.filename}",
            datetime.now().isoformat(),
            len(file_data.content),
            file_data.content,
            file_data.file_type
        ))
        
        conn.commit()
        file_id = cursor.lastrowid
        conn.close()
        
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        logger.error(f"Error processing text file: {e}")
        return {"status": "error", "message": str(e)}

def get_user_text_files(user_id):
    """Get all text files for a user"""
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, filename, description, upload_date, file_size, file_type
            FROM text_files 
            WHERE user_id = ? 
            ORDER BY upload_date DESC
        ''', (user_id,))
        
        files = []
        for row in cursor.fetchall():
            files.append({
                "id": row[0], "filename": row[1], "description": row[2],
                "upload_date": row[3], "file_size": row[4], "file_type": row[5]
            })
        conn.close()
        return files
    except Exception as e:
        logger.error(f"Error getting user text files: {e}")
        return []

def detect_language(question):
    """Detect if question is in Hindi, English, or mixed"""
    hindi_pattern = re.compile(r'[अ-ह]')
    has_hindi = bool(hindi_pattern.search(question))
    return "hindi" if has_hindi else "english"

def format_structured_answer(answer_text, is_hindi=False):
    """Convert AI response into structured HTML format with blue headings and SpeechSynthesis TTS"""
    
    # Clean the response
    answer_text = answer_text.strip()
    
    # Split into sections based on common heading patterns
    sections = []
    current_section = {"heading": "", "content": ""}
    
    lines = answer_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Detect headings for both Hindi and English
        if (line.endswith(':') or line.endswith('?') or
            (len(line) < 60 and any(keyword in line.lower() for keyword in 
                                   ['what is', 'definition', 'key', 'types', 'examples', 'formula', 'units', 
                                    'requirements', 'characteristics', 'advantages', 'disadvantages',
                                    'परिभाषा', 'प्रकार', 'उदाहरण', 'फॉर्मूला', 'इकाई', 'आवश्यकताएं', 
                                    'विशेषताएं', 'फायदे', 'नुकसान', 'महत्वपूर्ण', 'प्रक्रिया', 'कार्य',
                                    'क्या है', 'मतलब']))):
            
            # Save previous section if exists
            if current_section["heading"]:
                sections.append(current_section.copy())
            
            # Start new section
            current_section = {"heading": line.replace('**', '').replace('*', '').replace(':', '').replace('?', '').strip(), "content": ""}
        else:
            # Add to current section content
            if line:
                current_section["content"] += line + " "
    
    # Add the last section
    if current_section["heading"]:
        sections.append(current_section)
    
    # If no sections detected, create a default structure
    if not sections:
        sections = [{"heading": "उत्तर" if is_hindi else "Answer", "content": answer_text}]
    
    # Generate HTML with blue headings and TTS button
    tts_button = '''
    <div style="margin: 15px 0; text-align: center;">
        <button onclick="speakAnswer()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 14px;">
            🔊 सुनें / Listen
        </button>
        <button onclick="stopSpeaking()" style="background: #dc2626; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 14px; margin-left: 10px;">
            ⏹️ रोकें / Stop
        </button>
        <span id="tts-status" style="margin-left: 10px; font-size: 12px; color: #666;"></span>
    </div>
    '''
    
    html_output = tts_button
    
    for section in sections:
        if section["heading"]:
            html_output += f'<h3 style="color: #2563eb; font-weight: bold; margin-bottom: 8px;">{section["heading"]}</h3>'
        
        if section["content"]:
            # Format lists and bullet points
            content = section["content"].strip()
            
            # Convert markdown-style lists to HTML
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            
            # Handle bullet points
            lines = content.split('. ')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or line.startswith('•') or ' - ' in line[:20]:
                    line_content = line.lstrip("-*• ")
                    formatted_lines.append(f'<li>{line_content}</li>')
                else:
                    formatted_lines.append(line)
            
            if any('<li>' in line or line.startswith('<li>') for line in formatted_lines):
                html_output += '<ul style="margin-top: 0; margin-bottom: 12px;">'
                for line in formatted_lines:
                    if line.startswith('<li>'):
                        html_output += line
                    else:
                        if line and not line.startswith('<'):
                            html_output += f'<li>{line}</li>'
                html_output += '</ul>'
            else:
                html_output += f'<p style="margin-top: 0; margin-bottom: 12px;">{content}</p>'
    
    # Add JavaScript for SpeechSynthesis TTS functionality
    html_output += '''
    <script>
    let currentUtterance = null;
    
    function speakAnswer() {
        const answerDiv = document.getElementById('pappu-ai-answer');
        const statusSpan = document.getElementById('tts-status');
        const answerText = answerDiv.innerText || answerDiv.textContent;
        
        // Remove the TTS button text from the speech content
        const speechText = answerText.replace('सुनें / Listen', '')
                                    .replace('रोकें / Stop', '')
                                    .replace('Loading...', '')
                                    .replace('TTS Feature Currently Unavailable', '')
                                    .trim();
        
        // Stop any ongoing speech
        if (currentUtterance) {
            window.speechSynthesis.cancel();
        }
        
        // Check if SpeechSynthesis is supported
        if (!('speechSynthesis' in window)) {
            statusSpan.textContent = 'Speech synthesis not supported';
            return;
        }
        
        statusSpan.textContent = 'Speaking...';
        
        // Create new utterance
        currentUtterance = new SpeechSynthesisUtterance(speechText);
        
        // Auto-detect language and set voice properties
        const hasHindi = /[अ-ह]/.test(speechText);
        
        if (hasHindi) {
            // For Hindi text, try to use Hindi voice
            currentUtterance.lang = 'hi-IN';
            currentUtterance.rate = 0.9;
            currentUtterance.pitch = 1;
        } else {
            // For English text
            currentUtterance.lang = 'en-US';
            currentUtterance.rate = 1;
            currentUtterance.pitch = 1;
        }
        
        // Event handlers
        currentUtterance.onstart = function() {
            statusSpan.textContent = 'Speaking...';
        };
        
        currentUtterance.onend = function() {
            statusSpan.textContent = 'Completed';
            currentUtterance = null;
            setTimeout(() => {
                statusSpan.textContent = '';
            }, 2000);
        };
        
        currentUtterance.onerror = function(event) {
            console.error('SpeechSynthesis Error:', event);
            statusSpan.textContent = 'Error: ' + event.error;
            currentUtterance = null;
        };
        
        // Start speaking
        window.speechSynthesis.speak(currentUtterance);
    }
    
    function stopSpeaking() {
        const statusSpan = document.getElementById('tts-status');
        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
            statusSpan.textContent = 'Stopped';
            currentUtterance = null;
            setTimeout(() => {
                statusSpan.textContent = '';
            }, 2000);
        }
    }
    
    // Stop speech when page is unloaded
    window.addEventListener('beforeunload', function() {
        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }
    });
    </script>
    '''
    
    return html_output

def clean_text_for_tts(text):
    """Clean text for TTS by removing HTML tags and special formatting"""
    clean_text = re.sub(r'<[^>]*>', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = re.sub(r'[^\w\sअ-ह\.\,\?\!]', '', clean_text)
    return clean_text.strip()

# Routes
@app.get("/")
async def root():
    return {
        "status": "active", 
        "service": "Study AI - SpeechSynthesis TTS",
        "version": "4.0",
        "features": ["PDF Processing", "Text File Support", "Pure Hindi Answers", "SpeechSynthesis TTS", "Enhanced Security"],
        "tts_type": "browser_speech_synthesis",
        "security": "api_key_required"
    }

@app.get("/health")
async def health_check(request: Request):
    health_status = {
        "status": "healthy",
        "service": "Study AI - SpeechSynthesis TTS",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0"
    }
    
    # Check database
    try:
        conn = sqlite3.connect(settings.database_url)
        conn.close()
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = "disconnected"
        health_status["status"] = "degraded"
        logger.error(f"Database health check failed: {e}")
    
    # Check Gemini API
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        health_status["gemini_api"] = "available"
    except Exception as e:
        health_status["gemini_api"] = "unavailable"
        health_status["status"] = "degraded"
        logger.error(f"Gemini API health check failed: {e}")
    
    # System info
    health_status["rate_limiting"] = "disabled"  # Removed for simplicity
    health_status["authentication"] = "required"
    health_status["file_validation"] = "enabled"
    
    await log_usage("system", "health_check", request)
    return health_status

@app.post("/ask")
async def ask_question(request: Request, question_request: QuestionRequest):
    await log_usage(question_request.user_id, "ask", request)
    
    if not settings.gemini_api_key:
        raise CustomHTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        context = ""
        sources_used = 0
        detected_language = question_request.language
        
        # Auto-detect language if not specified
        if question_request.language == "auto":
            detected_language = detect_language(question_request.question)
        
        if question_request.use_pdf_context:
            relevant_sections = find_relevant_text_simple(question_request.question, question_request.user_id)
            
            if relevant_sections:
                context_chunks = []
                for section in relevant_sections:
                    context_chunks.append(f"From {section['filename']}: {section['text']}")
                
                context = "\n\nRelevant information from your files:\n" + "\n---\n".join(context_chunks)
                sources_used = len(relevant_sections)
        
        # Enhanced prompt for pure Hindi with English technical terms
        if detected_language == "hindi":
            prompt = f"""
            तुम एक helpful study assistant हो। निम्नलिखित प्रश्न का उत्तर पूरी तरह से हिंदी में दो।

            प्रश्न: {question_request.question}

            {context}

            **बहुत महत्वपूर्ण निर्देश:**
            1. पूरा उत्तर केवल हिंदी में लिखो
            2. तकनीकी/वैज्ञानिक शब्दों के लिए: पहले हिंदी शब्द लिखो, फिर कोष्ठक में अंग्रेजी शब्द लिखो
            3. फॉर्मेट: प्रकाश संश्लेषण (Photosynthesis), कार्बन डाइऑक्साइड (Carbon Dioxide), जल (Water)
            4. उत्तर को अच्छी तरह से संरचित करो - परिभाषा, मुख्य बिंदु, उदाहरण के अनुभागों में
            5. बुलेट पॉइंट्स का उपयोग करो
            6. महत्वपूर्ण शब्दों को बोल्ड करो
            7. उत्तर छात्रों के लिए आसानी से समझने योग्य बनाओ

            **उदाहरण फॉर्मेट:**
            **प्रकाश संश्लेषण (Photosynthesis) क्या है?**
            प्रकाश संश्लेषण (Photosynthesis) पौधों की वह प्रक्रिया है जिसमें...
            
            **मुख्य चरण:**
            • प्रकाश ऊर्जा (Light Energy) का अवशोषण
            • कार्बन डाइऑक्साइड (Carbon Dioxide) का उपयोग
            • ग्लूकोज (Glucose) का निर्माण

            उत्तर:
            """
        else:
            prompt = f"""
            You are a helpful study assistant. Answer the following question in English with clear structure.

            QUESTION: {question_request.question}

            {context}

            Please provide a comprehensive answer with:
            - Clear definition/overview section
            - Key points in organized sections
            - Bullet points for lists
            - Bold text for important terms
            - Simple example if applicable
            - Make it student-friendly and easy to understand

            Use proper section headings to organize the content.

            ANSWER:
            """
        
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        response = model.generate_content(prompt)
        
        # Format the answer with structured HTML and SpeechSynthesis TTS
        formatted_answer = format_structured_answer(response.text, is_hindi=(detected_language == "hindi"))
        
        return {
            "question": question_request.question,
            "answer": formatted_answer,
            "raw_answer": response.text,
            "clean_text": clean_text_for_tts(response.text),
            "status": "success",
            "context_used": sources_used > 0,
            "sources_used": sources_used,
            "search_method": "keyword_based",
            "format": "structured_html",
            "detected_language": detected_language,
            "language_used": "hindi" if detected_language == "hindi" else "english",
            "tts_type": "browser_speech_synthesis"
        }
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise CustomHTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Text File Processing Endpoints
@app.post("/process-text-file")
async def process_text_file(request: Request, file_data: TextFileRequest, auth: bool = Depends(verify_api_key)):
    """Process text files from WordPress media library"""
    await log_usage(file_data.user_id, "process_text_file", request)
    
    try:
        result = process_text_file_content(file_data)
        
        if result["status"] == "duplicate":
            return {
                "status": "duplicate", 
                "message": "File already processed",
                "file_id": result["file_id"]
            }
        elif result["status"] == "success":
            return {
                "status": "success",
                "message": "Text file processed successfully",
                "file_id": result["file_id"],
                "filename": file_data.filename,
                "word_count": len(file_data.content.split()),
                "file_type": file_data.file_type
            }
        else:
            raise CustomHTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"Error processing text file: {e}")
        raise CustomHTTPException(status_code=500, detail=f"Error processing text file: {str(e)}")

@app.get("/user-text-files/{user_id}")
async def get_user_text_files_list(request: Request, user_id: str, auth: bool = Depends(verify_api_key)):
    """Get all text files for a user"""
    await log_usage(user_id, "get_user_text_files", request)
    
    try:
        files = get_user_text_files(user_id)
        return {
            "user_id": user_id,
            "files": files,
            "total_files": len(files)
        }
    except Exception as e:
        logger.error(f"Error getting user text files: {e}")
        raise CustomHTTPException(status_code=500, detail=f"Error: {str(e)}")

# PDF endpoints with enhanced security
@app.post("/upload-pdf")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    description: str = Form(""),
    auth: bool = Depends(verify_api_key)
):
    await log_usage(user_id, "upload_pdf", request)
    
    if not file.filename.endswith('.pdf'):
        raise CustomHTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_content = await file.read()
    file_size = len(file_content)
    
    # Validate file size
    if file_size > settings.max_file_size:
        raise CustomHTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
    
    # Validate file type
    if not validate_pdf_file(file_content):
        raise CustomHTTPException(status_code=400, detail="Invalid PDF file")
    
    file_hash = hashlib.md5(file_content).hexdigest()
    
    pdf_id = add_pdf_to_db(file.filename, file_hash, user_id, description, file_size)
    
    if pdf_id is None:
        return {"status": "duplicate", "message": "PDF already uploaded"}
    
    text_content = extract_text_from_pdf(file_content)
    word_count = len(text_content.split()) if text_content else 0
    
    processed = False
    if text_content:
        processed = process_pdf_content(pdf_id, text_content)
    
    return {
        "status": "success",
        "message": "PDF uploaded successfully",
        "pdf_id": pdf_id,
        "filename": file.filename,
        "word_count": word_count,
        "description": description,
        "processed": processed,
        "context_search_available": True
    }

@app.get("/user-pdfs/{user_id}")
async def get_user_pdfs_list(request: Request, user_id: str, auth: bool = Depends(verify_api_key)):
    await log_usage(user_id, "get_user_pdfs", request)
    
    try:
        pdfs = get_user_pdfs(user_id)
        return {
            "user_id": user_id,
            "pdfs": pdfs,
            "total_pdfs": len(pdfs),
            "processed_pdfs": sum(1 for p in pdfs if p['processed'])
        }
    except Exception as e:
        logger.error(f"Error getting user PDFs: {e}")
        raise CustomHTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/ask-simple")
async def ask_simple(
    request: Request,
    question: str, 
    user_id: str = "default", 
    use_context: bool = True, 
    language: str = "auto"
):
    """Simple GET endpoint for WordPress plugin - Returns structured answers with SpeechSynthesis TTS"""
    await log_usage(user_id, "ask_simple", request)
    
    if not settings.gemini_api_key:
        return {"error": "Gemini API key not configured"}
    try:
        question_request = QuestionRequest(
            question=question, 
            user_id=user_id, 
            use_pdf_context=use_context,
            language=language
        )
        return await ask_question(request, question_request)
    except Exception as e:
        logger.error(f"Error in ask_simple: {e}")
        return {"error": str(e)}

@app.delete("/delete-pdf/{pdf_id}")
async def delete_pdf(request: Request, pdf_id: int, user_id: str, auth: bool = Depends(verify_api_key)):
    """Delete a PDF and its content"""
    await log_usage(user_id, "delete_pdf", request)
    
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM pdf_files WHERE id = ? AND user_id = ?', (pdf_id, user_id))
        pdf_info = cursor.fetchone()
        
        if not pdf_info:
            raise CustomHTTPException(status_code=404, detail="PDF not found")
        
        cursor.execute('DELETE FROM pdf_files WHERE id = ?', (pdf_id,))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "PDF deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting PDF: {e}")
        raise CustomHTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

@app.delete("/delete-text-file/{file_id}")
async def delete_text_file(request: Request, file_id: int, user_id: str, auth: bool = Depends(verify_api_key)):
    """Delete a text file and its content"""
    await log_usage(user_id, "delete_text_file", request)
    
    try:
        conn = sqlite3.connect(settings.database_url)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM text_files WHERE id = ? AND user_id = ?', (file_id, user_id))
        file_info = cursor.fetchone()
        
        if not file_info:
            raise CustomHTTPException(status_code=404, detail="Text file not found")
        
        cursor.execute('DELETE FROM text_files WHERE id = ?', (file_id,))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Text file deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting text file: {e}")
        raise CustomHTTPException(status_code=500, detail=f"Error deleting text file: {str(e)}")

# TTS Support Check Endpoint
@app.get("/tts-support")
async def check_tts_support(request: Request):
    """Check TTS support information"""
    await log_usage("system", "tts_support", request)
    
    return {
        "tts_type": "browser_speech_synthesis",
        "supported_languages": ["hindi", "english"],
        "requires_browser_support": True,
        "features": ["auto_language_detection", "voice_selection", "playback_control"],
        "browser_requirements": ["SpeechSynthesis API support"]
    }

# Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
