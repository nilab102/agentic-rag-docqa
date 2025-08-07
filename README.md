# Agentic RAG Document Q&A

An advanced agentic RAG (Retrieval-Augmented Generation) application using LlamaIndex and Google Gemini for intelligent chat-based question answering over uploaded documents. Features accurate answers with citations (file name + page number), modern web UI, and unified backend architecture.

## 🚀 Features

- **Document Processing**: Upload and process PDFs, Word documents, and Excel files
- **Intelligent Q&A**: Advanced RAG-powered question answering with context-aware responses
- **Accurate Citations**: Source citations with file names and page numbers
- **Modern Web UI**: Beautiful, responsive interface for document management and queries
- **Real-time Processing**: Background task processing with progress tracking
- **Vector Storage**: ChromaDB-based vector storage for efficient document retrieval
- **Google Gemini Integration**: State-of-the-art LLM and embedding models

## 🏗️ Architecture

This is a **unified server application** that combines:
- **Backend**: FastAPI server with comprehensive REST API
- **Frontend**: Modern HTML/CSS/JavaScript web interface
- **Vector Database**: ChromaDB for document embeddings
- **Document Processing**: LlamaIndex pipeline for chunking and indexing
- **AI Models**: Google Gemini for LLM and embeddings

## 📋 Requirements

- **Python**: 3.10 or higher
- **Google API Key**: For Gemini LLM and embeddings
- **Optional**: PostgreSQL database (can use SQLite for development)

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd agentic-rag-docqa
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
cp env.example .env
```

Edit `.env` file with your configuration:
```env
# Required: Google API Key
GOOGLE_API_KEY="your_google_api_key_here"

# Optional: Database configuration
DB_HOST=localhost
DB_PORT=5433
DB_NAME=document_db_test
DB_USER=postgres
DB_PASSWORD=your_db_password_here

# Vector store settings
CHROMA_PERSIST_DIR=./chroma_db

# Model configuration
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_LLM_MODEL=gemini-2.5-flash

# UI settings
UI_PORT=1234
```

### 5. Start the Application
```bash
python start.py
```

The application will start on `http://localhost:1234` (or the port specified in UI_PORT).

## 🎯 Quick Start

1. **Start the server**: `python start.py`
2. **Open browser**: Navigate to `http://localhost:1234`
3. **Upload documents**: Use the web interface to upload PDFs, Word docs, or Excel files
4. **Ask questions**: Use the chat interface to query your documents
5. **View citations**: See source files and page numbers for each answer

## 📚 API Endpoints

### Core Endpoints
- `GET /` - Web UI interface
- `GET /health` - System health check
- `GET /docs` - Interactive API documentation

### Document Management
- `POST /upload` - Upload and process documents
- `GET /files` - List all uploaded files
- `DELETE /files/{filename}` - Delete a specific file

### Task Management
- `GET /tasks` - List all processing tasks
- `GET /tasks/{task_id}` - Get task status and progress
- `DELETE /tasks/{task_id}` - Delete a task

### Query Interface
- `POST /query` - Query documents with RAG-powered responses

## 🔧 Project Structure

```
agentic-rag-docqa/
├── start.py              # Main application entry point
├── main.py               # Core application logic
├── serve_ui.py           # UI serving utilities
├── index.html            # Web interface
├── requirements.txt      # Python dependencies
├── env.example           # Environment configuration template
├── uploads/              # Temporary file uploads
├── chroma_db/            # Vector database storage
└── utilites/             # Core processing modules
    ├── pipline_upload.py # Document processing pipeline
    ├── query.py          # Query engine and RAG logic
    ├── vector_storage.py # Vector database operations
    ├── parser.py         # Document parsing utilities
    └── chunker.py        # Text chunking algorithms
```

## 🎨 Web Interface Features

- **Modern Design**: Beautiful gradient UI with responsive layout
- **Document Upload**: Drag-and-drop file upload with progress tracking
- **Real-time Processing**: Live progress updates for document processing
- **Chat Interface**: Interactive Q&A with document context
- **File Management**: View, search, and delete uploaded documents
- **Task Monitoring**: Track processing tasks and their status

## 🔍 Advanced Features

### Document Processing Pipeline
- **Multi-format Support**: PDF, Word, Excel documents
- **Intelligent Chunking**: Context-aware text segmentation
- **Metadata Extraction**: File information and page numbers
- **Background Processing**: Asynchronous document processing

### RAG Query Engine
- **Semantic Search**: Vector similarity search for relevant content
- **Context Retrieval**: Intelligent context selection
- **Citation Generation**: Automatic source attribution
- **Response Generation**: LLM-powered answer synthesis

### Vector Storage
- **ChromaDB Integration**: Efficient vector database
- **Embedding Management**: Google Gemini embeddings
- **Index Optimization**: Fast retrieval and similarity search

## 🚀 Deployment

### Development
```bash
python start.py
```

### Production with Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 1234

CMD ["python", "start.py"]
```

### Environment Variables for Production
- Set `GOOGLE_API_KEY` with your production API key
- Configure database connection for PostgreSQL
- Set appropriate `UI_PORT` for your deployment
- Configure `CHROMA_PERSIST_DIR` for persistent storage

## 🔧 Configuration Options

### Model Settings
- `GEMINI_LLM_MODEL`: LLM model (default: gemini-2.5-flash)
- `GEMINI_EMBEDDING_MODEL`: Embedding model (default: models/embedding-001)
- `TEMPERATURE`: Response creativity (0.0-1.0)
- `MAX_TOKENS`: Maximum response length

### Processing Settings
- `EMBEDDING_DIMENSION`: Vector embedding dimensions
- `SIMILARITY_TOP_K`: Number of similar chunks to retrieve
- `SIMILARITY_THRESHOLD`: Similarity threshold for retrieval

### Database Settings
- `DB_HOST`, `DB_PORT`, `DB_NAME`: PostgreSQL configuration
- `CHROMA_PERSIST_DIR`: Vector database storage location

## 🐛 Troubleshooting

### Common Issues

1. **Google API Key Error**
   - Ensure `GOOGLE_API_KEY` is set in `.env`
   - Verify API key has access to Gemini models

2. **Port Already in Use**
   - Change `UI_PORT` in `.env` file
   - Or kill existing process on the port

3. **Document Processing Fails**
   - Check file format is supported (PDF, DOCX, XLSX)
   - Verify file is not corrupted
   - Check logs for specific error messages

4. **Database Connection Issues**
   - Verify PostgreSQL is running (if using)
   - Check database credentials in `.env`
   - ChromaDB will use local storage if no database configured

### Logs and Debugging
- Application logs are displayed in the console
- Check `chroma_db/` directory for vector database files
- Monitor `uploads/` directory for temporary files

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs` when server is running
- Open an issue on the repository

---

**Built with ❤️ using FastAPI, LlamaIndex, and Google Gemini**


