#!/usr/bin/env python3
"""
Startup script for the Unified Agentic RAG Document Q&A Server
This script provides a simple way to start the server with proper error handling.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required files exist."""
    required_files = [
        "unified_server.py",
        "index.html",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all files are in the current directory.")
        return False
    
    return True

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install required packages if needed."""
    try:
        print("📦 Checking dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_server():
    """Start the unified server."""
    port = int(os.environ.get("UI_PORT", 8000))
    
    print(f"🚀 Starting Unified Agentic RAG Document Q&A Server...")
    print(f"📁 Port: {port}")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"⏳ Starting server...")
    
    try:
        # Start the server
        subprocess.run([sys.executable, "unified_server.py"], check=True)
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("=" * 60)
    print("🤖 Agentic RAG Document Q&A - Unified Server")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required files
    if not check_dependencies():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Start server
    print("\n" + "=" * 60)
    start_server()

if __name__ == "__main__":
    main() 