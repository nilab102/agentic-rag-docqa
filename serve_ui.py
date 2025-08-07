#!/usr/bin/env python3
"""
Simple HTTP server to serve the RAG Document Q&A UI
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = int(os.environ.get("UI_PORT", 8000))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Change to the directory containing this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if index.html exists
    if not Path('index.html').exists():
        print("❌ Error: index.html not found in the current directory")
        return
    
    # Create server
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🚀 UI Server started at http://localhost:{PORT}")
        print(f"📁 Serving files from: {script_dir}")
        print(f"🌐 Opening browser automatically...")
        
        # Open browser
        webbrowser.open(f'http://localhost:{PORT}')
        
        try:
            print(f"✅ Server running. Press Ctrl+C to stop.")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n🛑 Server stopped.")

if __name__ == "__main__":
    main() 