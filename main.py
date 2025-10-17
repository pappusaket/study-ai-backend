from fastapi.responses import HTMLResponse

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
                text-align: left;
                display: inline-block;
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
            a:hover {
                background: rgba(0,0,0,0.5);
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
                <strong>API Endpoints:</strong>
                <a href="/health">🔍 Health Check</a>
                <a href="/docs">📚 API Documentation</a>
                <a href="/test">🧪 Test Endpoint</a>
            </div>
            
            <div style="margin-top: 20px; font-size: 12px; opacity: 0.8;">
                Last checked: <span id="timestamp">loading...</span>
            </div>
        </div>

        <script>
            // Live timestamp
            function updateTime() {
                document.getElementById('timestamp').textContent = new Date().toLocaleString();
            }
            updateTime();
            setInterval(updateTime, 1000);
            
            // Live status check
            async function checkStatus() {
                try {
                    const response = await fetch('/health');
                    if (response.ok) {
                        console.log('✅ Server is healthy');
                    }
                } catch (error) {
                    console.log('❌ Server connection issue');
                }
            }
            // Check every 30 seconds
            setInterval(checkStatus, 30000);
        </script>
    </body>
    </html>
    """
