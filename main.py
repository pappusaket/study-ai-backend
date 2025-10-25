from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
import tempfile
import base64

# TTS setup
TTS_AVAILABLE = False
TTS_PROVIDER = "none"

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
    TTS_PROVIDER = "gtts"
    print("✓ gTTS loaded successfully")
except ImportError as e:
    print(f"✗ gTTS not available: {e}")
    TTS_AVAILABLE = False
    TTS_PROVIDER = "none"

app = FastAPI(title="Study AI - Hindi+English Answers")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)

# Database setup
DB_PATH = os.path.join(os.getcwd(), "study_ai.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
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
    
    conn.commit()
    conn.close()

init_db()

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

class TTSRequest(BaseModel):
    text: str
    language: str = "auto"

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
        raise HTTPException(status_code=400, detail=f"PDF reading error: {str(e)}")

def find_relevant_text_simple(question, user_id):
    """Simple keyword-based search for relevant text"""
    conn = sqlite3.connect(DB_PATH)
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

def process_pdf_content(pdf_id, text_content):
    """Store PDF text content"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE pdf_files SET processed = TRUE, text_content = ? WHERE id = ?',
            (text_content, pdf_id)
        )
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error storing PDF content: {e}")
        return False

def add_pdf_to_db(filename, file_hash, user_id, description, file_size):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO pdf_files 
            (filename, file_hash, user_id, description, upload_date, status, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, file_hash, user_id, description, datetime.now().isoformat(), 'uploaded', file_size))
        conn.commit()
        pdf_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        cursor.execute('SELECT id FROM pdf_files WHERE file_hash = ?', (file_hash,))
        result = cursor.fetchone()
        pdf_id = result[0] if result else None
    finally:
        conn.close()
    return pdf_id

def get_user_pdfs(user_id):
    conn = sqlite3.connect(DB_PATH)
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

def process_text_file_content(file_data: TextFileRequest):
    """Process and store text file content from WordPress"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        file_hash = hashlib.md5(file_data.content.encode()).hexdigest()
        
        cursor.execute('SELECT id FROM text_files WHERE file_hash = ? AND user_id = ?', 
                      (file_hash, file_data.user_id))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return {"status": "duplicate", "file_id": existing[0]}
        
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
        print(f"Error processing text file: {e}")
        return {"status": "error", "message": str(e)}

def get_user_text_files(user_id):
    """Get all text files for a user"""
    conn = sqlite3.connect(DB_PATH)
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

def detect_language(question):
    """Detect if question is in Hindi, English, or mixed"""
    hindi_pattern = re.compile(r'[अ-ह]')
    
    has_hindi = bool(hindi_pattern.search(question))
    
    if has_hindi:
        return "hindi"
    else:
        return "english"

def format_structured_answer(answer_text, is_hindi=False):
    """Convert AI response into structured HTML format with blue headings and TTS button"""
    
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
    tts_button = ''
    if TTS_AVAILABLE:
        tts_button = '''
        <div style="margin: 15px 0; text-align: center;">
            <button onclick="speakAnswer()" style="background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 14px;">
                🔊 सुनें / Listen
            </button>
            <span id="tts-status" style="margin-left: 10px; font-size: 12px; color: #666;"></span>
        </div>
        '''
    else:
        tts_button = '''
        <div style="margin: 15px 0; text-align: center; color: #666; font-size: 12px;">
            🔊 TTS Feature Currently Unavailable
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
    
    # Add JavaScript for TTS functionality
    if TTS_AVAILABLE:
        html_output += '''
        <script>
        function speakAnswer() {
            const answerDiv = document.getElementById('pappu-ai-answer');
            const statusSpan = document.getElementById('tts-status');
            const answerText = answerDiv.innerText || answerDiv.textContent;
            
            statusSpan.textContent = 'Loading...';
            
            // Remove the TTS button text from the speech content
            const speechText = answerText.replace('सुनें / Listen', '').replace('Loading...', '').replace('TTS Feature Currently Unavailable', '').trim();
            
            fetch('https://study-ai-backend-fylo.onrender.com/text-to-speech', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: speechText,
                    language: 'auto'
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.audio_base64) {
                    statusSpan.textContent = 'Playing...';
                    
                    // Create audio from base64
                    const audio = new Audio('data:audio/mp3;base64,' + data.audio_base64);
                    audio.play();
                    
                    audio.onended = function() {
                        statusSpan.textContent = 'Completed';
                        setTimeout(() => {
                            statusSpan.textContent = '';
                        }, 2000);
                    };
                    
                    audio.onerror = function() {
                        statusSpan.textContent = 'Audio error';
                    };
                } else {
                    statusSpan.textContent = 'Error: ' + (data.error || 'Unknown error');
                }
            })
            .catch(error => {
                console.error('TTS Error:', error);
                statusSpan.textContent = 'Error: ' + error.message;
            });
        }
        </script>
        '''
    
    return html_output

def clean_text_for_tts(text):
    """Clean text for TTS by removing HTML tags and special formatting"""
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]*>', '', text)
    # Remove extra spaces and newlines
    clean_text = re.sub(r'\s+', ' ', clean_text)
    # Remove special characters but keep Hindi and English text
    clean_text = re.sub(r'[^\w\sअ-ह\.\,\?\!]', '', clean_text)
    return clean_text.strip()

def create_tts_audio_base64(text, language="auto"):
    """Create TTS audio and return as base64 string"""
    if not TTS_AVAILABLE:
        return None, "TTS not available"
    
    try:
        # Clean the text
        clean_text = clean_text_for_tts(text)
        
        if len(clean_text) > 4000:
            clean_text = clean_text[:4000] + "..."
        
        # Determine language
        if language == "auto":
            has_hindi = re.search(r'[अ-ह]', clean_text)
            tts_language = "hi" if has_hindi else "en"
        else:
            tts_language = language
        
        if TTS_PROVIDER == "gtts":
            # Use gTTS and get audio as bytes
            tts = gTTS(text=clean_text, lang=tts_language, slow=False)
            
            # Save to bytes buffer instead of file
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
            return audio_base64, None
        else:
            return None, "No TTS provider available"
            
    except Exception as e:
        return None, f"TTS conversion failed: {str(e)}"

@app.get("/")
async def root():
    return {
        "status": "active", 
        "service": "Study AI - Hindi+English Answers",
        "version": "2.8",
        "features": ["PDF Processing", "Text File Support", "Pure Hindi Answers", "Structured Format"],
        "tts_available": TTS_AVAILABLE,
        "tts_provider": TTS_PROVIDER
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Study AI - Hindi+English Answers",
        "gemini": "configured",
        "database": "connected",
        "tts": TTS_PROVIDER,
        "tts_available": TTS_AVAILABLE,
        "languages": ["auto", "hindi", "english"],
        "features": ["Pure Hindi Q&A", "PDF Context", "Text File Support", "Blue Headings"]
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        context = ""
        sources_used = 0
        detected_language = request.language
        
        # Auto-detect language if not specified
        if request.language == "auto":
            detected_language = detect_language(request.question)
        
        if request.use_pdf_context:
            relevant_sections = find_relevant_text_simple(request.question, request.user_id)
            
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

            प्रश्न: {request.question}

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

            QUESTION: {request.question}

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
        
        # Format the answer with structured HTML and TTS button
        formatted_answer = format_structured_answer(response.text, is_hindi=(detected_language == "hindi"))
        
        return {
            "question": request.question,
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
            "tts_available": TTS_AVAILABLE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# TTS Endpoint - Now returns base64 audio directly
@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return base64 audio"""
    try:
        if not request.text or len(request.text.strip()) == 0:
            return {"error": "No text provided"}
        
        # Create TTS audio as base64
        audio_base64, error = create_tts_audio_base64(request.text, request.language)
        
        if error:
            return {"error": error}
        
        if audio_base64:
            return {
                "audio_base64": audio_base64,
                "text_length": len(request.text),
                "tts_provider": TTS_PROVIDER,
                "message": "Audio generated successfully"
            }
        else:
            return {"error": "TTS conversion failed"}
        
    except Exception as e:
        return {"error": f"TTS conversion failed: {str(e)}"}

# Text File Processing Endpoints
@app.post("/process-text-file")
async def process_text_file(request: TextFileRequest):
    """Process text files from WordPress media library"""
    try:
        result = process_text_file_content(request)
        
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
                "filename": request.filename,
                "word_count": len(request.content.split()),
                "file_type": request.file_type
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text file: {str(e)}")

@app.get("/user-text-files/{user_id}")
async def get_user_text_files_list(user_id: str):
    """Get all text files for a user"""
    try:
        files = get_user_text_files(user_id)
        return {
            "user_id": user_id,
            "files": files,
            "total_files": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Existing PDF endpoints
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...), description: str = Form("")):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    file_content = await file.read()
    file_size = len(file_content)
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
async def get_user_pdfs_list(user_id: str):
    try:
        pdfs = get_user_pdfs(user_id)
        return {
            "user_id": user_id,
            "pdfs": pdfs,
            "total_pdfs": len(pdfs),
            "processed_pdfs": sum(1 for p in pdfs if p['processed'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/ask-simple")
async def ask_simple(question: str, user_id: str = "default", use_context: bool = True, language: str = "auto"):
    """Simple GET endpoint for WordPress plugin - Returns structured answers"""
    if not GEMINI_API_KEY:
        return {"error": "Gemini API key not configured"}
    try:
        request = QuestionRequest(
            question=question, 
            user_id=user_id, 
            use_pdf_context=use_context,
            language=language
        )
        return await ask_question(request)
    except Exception as e:
        return {"error": str(e)}

@app.delete("/delete-pdf/{pdf_id}")
async def delete_pdf(pdf_id: int, user_id: str):
    """Delete a PDF and its content"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM pdf_files WHERE id = ? AND user_id = ?', (pdf_id, user_id))
        pdf_info = cursor.fetchone()
        
        if not pdf_info:
            raise HTTPException(status_code=404, detail="PDF not found")
        
        cursor.execute('DELETE FROM pdf_files WHERE id = ?', (pdf_id,))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "PDF deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

@app.delete("/delete-text-file/{file_id}")
async def delete_text_file(file_id: int, user_id: str):
    """Delete a text file and its content"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM text_files WHERE id = ? AND user_id = ?', (file_id, user_id))
        file_info = cursor.fetchone()
        
        if not file_info:
            raise HTTPException(status_code=404, detail="Text file not found")
        
        cursor.execute('DELETE FROM text_files WHERE id = ?', (file_id,))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Text file deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting text file: {str(e)}")

# Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
