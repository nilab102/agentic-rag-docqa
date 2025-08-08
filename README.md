# Agentic RAG Document Q&A

An advanced agentic RAG (Retrieval-Augmented Generation) application using LlamaIndex and Google Gemini for intelligent chat-based question answering over uploaded documents. Features accurate answers with citations (file name + page number), modern web UI, and unified backend architecture with multi-language support.

## 🚀 Features

- **Multi-Format Document Processing**: Upload and process PDFs, Word documents, Excel files, text files, Markdown, and RTF documents
- **Enterprise Document Conversion**: CloudConvert integration for 200+ additional formats (CAD, images, e-books, etc.)
- **Intelligent Q&A**: Advanced RAG-powered question answering with context-aware responses
- **Accurate Citations**: Source citations with file names and page numbers
- **Multi-Language Support**: Automatic language detection with Arabic and English support
- **Modern Web UI**: Beautiful, responsive interface for document management and queries
- **Real-time Processing**: Background task processing with progress tracking
- **Vector Storage**: ChromaDB-based vector storage for efficient document retrieval
- **Google Gemini Integration**: State-of-the-art LLM and embedding models
- **SSL Support**: Built-in HTTPS support for secure connections
- **Task Management**: Comprehensive task tracking and management system

## 🏗️ Architecture

This is a **unified server application** that combines:
- **Backend**: FastAPI server with comprehensive REST API
- **Frontend**: Modern HTML/CSS/JavaScript web interface
- **Vector Database**: ChromaDB for document embeddings and storage
- **Document Processing**: LlamaIndex pipeline for intelligent chunking and indexing
- **Document Conversion**: CloudConvert integration for enterprise-grade format support
- **AI Models**: Google Gemini for LLM and embeddings with advanced language support
- **Task Management**: Background processing with real-time progress tracking

## 📋 Requirements

- **Python**: 3.10 or higher
- **Google API Key**: For Gemini LLM and embeddings
- **Optional**: CloudConvert API key for enterprise-grade document conversion
- **SSL Certificates**: For HTTPS support (cert.pem and key.pem files)

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

# Google Gemini Models
GOOGLE_GEMINI_MODEL="gemini-2.5-flash"
google_gemini_name_light="gemini-2.5-flash"
google_gemini_embedding_name="models/gemini-embedding-exp-03-07"

# CloudConvert API (Optional - Enhanced Document Conversion)
# Get your API key from: https://cloudconvert.com/dashboard/api/v2/keys
# Enables conversion of additional file formats (CAD, images, etc.)
CLOUDCONVERT_API_KEY="your_cloudconvert_api_key_here"

# Vector store settings
CHROMA_PERSIST_DIR=./chroma_db

# Model configuration
GEMINI_EMBEDDING_MODEL=models/embedding-001
GEMINI_LLM_MODEL=gemini-2.5-flash

# API settings
API_PORT=1090

# Embedding & Similarity Settings
EMBEDDING_DIMENSION=1536
SIMILARITY_TOP_K=10
SIMILARITY_THRESHOLD=0.7

# LLM Output Settings
RESPONSE_MODE=compact
TEMPERATURE=0.1
MAX_TOKENS=2048
```

### 5. SSL Certificate Setup (Optional)
For HTTPS support, place your SSL certificates in the project root:
- `cert.pem` - SSL certificate file
- `key.pem` - SSL private key file

### 6. Start the Application
```bash
python main.py
```

The application will start on `https://localhost:1090` (or the port specified in API_PORT).

## 🎯 Quick Start

### Basic Setup
1. **Start the server**: `python main.py`
2. **Open browser**: Navigate to `https://localhost:1090`
3. **Upload documents**: Use the web interface to upload supported files
4. **Ask questions**: Use the chat interface to query your documents
5. **View citations**: See source files and page numbers for each answer
6. **Monitor tasks**: Track document processing progress in real-time

### Enterprise Setup (with CloudConvert)
1. **Get CloudConvert API key**: Visit [CloudConvert Dashboard](https://cloudconvert.com/dashboard/api/v2/keys)
2. **Configure environment**: Add `CLOUDCONVERT_API_KEY` to your `.env` file
3. **Upload any format**: Support for 200+ file formats including CAD, images, e-books
4. **Automatic conversion**: Files are converted and processed seamlessly
5. **Query converted documents**: Use the same RAG interface for all document types

## 📚 API Endpoints

### Core Endpoints
- `GET /` - Web UI interface
- `GET /api` - API information
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
├── main.py               # Main application entry point and FastAPI server
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
- **Multi-language Support**: Automatic language detection and response

## 🔍 Advanced Features

### Document Processing Pipeline
- **Multi-format Support**: PDF, DOCX, DOC, TXT, MD, RTF documents
- **Enterprise Conversion**: CloudConvert integration for 200+ additional formats (CAD, images, etc.)
- **Intelligent Chunking**: Context-aware text segmentation with page boundaries
- **Metadata Extraction**: File information, page numbers, and document structure
- **Background Processing**: Asynchronous document processing with progress tracking
- **Advanced Parsing**: Smart document processor with table and structure recognition

### RAG Query Engine
- **Semantic Search**: Vector similarity search for relevant content
- **Context Retrieval**: Intelligent context selection with metadata filtering
- **Citation Generation**: Automatic source attribution with page numbers
- **Response Generation**: LLM-powered answer synthesis
- **Language Detection**: Automatic detection and response in Arabic, English, and other languages
- **Page Citations**: Automatic page number extraction and citation

### Vector Storage
- **ChromaDB Integration**: Efficient vector database with persistent storage
- **Embedding Management**: Google Gemini embeddings with configurable dimensions
- **Index Optimization**: Fast retrieval and similarity search
- **Metadata Filtering**: Advanced filtering by file, page, date, and content

### Task Management System
- **Background Processing**: Non-blocking document processing
- **Progress Tracking**: Real-time progress updates with detailed status
- **Task Persistence**: Task information storage and retrieval
- **Error Handling**: Comprehensive error tracking and reporting

## 🚀 Deployment

### Development
```bash
python main.py
```

### Production with Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 1090

CMD ["python", "main.py"]
```

### Environment Variables for Production
- Set `GOOGLE_API_KEY` with your production API key
- Configure `CHROMA_PERSIST_DIR` for persistent storage
- Set appropriate `API_PORT` for your deployment
- Add SSL certificates for HTTPS support
- **Optional**: Set `CLOUDCONVERT_API_KEY` for enterprise document conversion (200+ formats)

## 🔧 Configuration Options

### Model Settings
- `GEMINI_LLM_MODEL`: LLM model (default: gemini-2.5-flash)
- `GEMINI_EMBEDDING_MODEL`: Embedding model (default: models/embedding-001)
- `google_gemini_embedding_name`: Advanced embedding model (default: models/gemini-embedding-exp-03-07)
- `TEMPERATURE`: Response creativity (0.0-1.0)
- `MAX_TOKENS`: Maximum response length

### Processing Settings
- `EMBEDDING_DIMENSION`: Vector embedding dimensions (auto-detected from model)
- `SIMILARITY_TOP_K`: Number of similar chunks to retrieve
- `SIMILARITY_THRESHOLD`: Similarity threshold for retrieval

### Vector Database Settings
- `CHROMA_PERSIST_DIR`: Vector database storage location
- Collection name and configuration managed automatically

### Server Settings
- `API_PORT`: Server port (default: 1090)
- SSL certificates for HTTPS support

## 🔄 Enterprise Document Conversion (CloudConvert)

### Overview
The application integrates with **CloudConvert**, a leading enterprise document conversion service, to support 200+ file formats beyond the standard document types.

### Supported Formats with CloudConvert
- **CAD Files**: AutoCAD (.dwg, .dxf), SketchUp (.skp), SolidWorks (.sldprt)
- **Image Formats**: TIFF, PNG, JPEG, BMP, WebP, SVG
- **Office Documents**: Legacy formats (.xls, .ppt, .pps), OpenDocument formats
- **E-books**: EPUB, MOBI, AZW3
- **Web Formats**: HTML, MHTML, WebP
- **Vector Graphics**: AI, EPS, CDR
- **And 200+ more formats...**

### Setup Instructions

#### 1. Get CloudConvert API Key
1. Visit [CloudConvert Dashboard](https://cloudconvert.com/dashboard/api/v2/keys)
2. Sign up for a free account or use existing credentials
3. Generate a new API key with appropriate permissions
4. Copy the API key to your `.env` file

#### 2. Configure Environment
```env
# Add to your .env file
CLOUDCONVERT_API_KEY="your_cloudconvert_api_key_here"
```

#### 3. Usage
- Upload any supported format through the web interface
- The system automatically detects format and converts if needed
- Converted documents are processed through the standard RAG pipeline
- Original and converted files are managed seamlessly

### Pricing & Limits
- **Free Tier**: 25 conversions/day, 1GB storage
- **Paid Plans**: Starting at $9/month for 1,000 conversions
- **Enterprise**: Custom pricing for high-volume usage

### Benefits
- **Universal Format Support**: Handle virtually any document type
- **High-Quality Conversion**: Enterprise-grade conversion engine
- **Seamless Integration**: Works transparently with existing workflow
- **Scalable**: Pay-as-you-go pricing model

## 🌍 Multi-Language Support

The application automatically detects and responds in multiple languages:

### Arabic Support
- Automatic Arabic text detection
- Arabic-specific RAG prompts
- Proper Arabic formatting and citations
- Page number citations in Arabic format

### English Support
- Standard English processing
- English-specific prompts and formatting
- Page number citations in English format

### Other Languages
- Generic language support
- Maintains original language in responses
- Appropriate formatting for various languages

## 🐛 Troubleshooting

### Common Issues

1. **Google API Key Error**
   - Ensure `GOOGLE_API_KEY` is set in `.env`
   - Verify API key has access to Gemini models

2. **Port Already in Use**
   - Change `API_PORT` in `.env` file
   - Or kill existing process on the port

3. **SSL Certificate Issues**
   - Ensure `cert.pem` and `key.pem` files exist in project root
   - Verify certificate files are valid and readable
   - For development, you can modify main.py to run without SSL

4. **Document Processing Fails**
   - Check file format is supported (PDF, DOCX, DOC, TXT, MD, RTF)
   - Verify file is not corrupted
   - Check logs for specific error messages
   - Monitor task status via `/tasks/{task_id}` endpoint

5. **Vector Database Issues**
   - Check `CHROMA_PERSIST_DIR` exists and is writable
   - Verify ChromaDB is properly initialized
   - Check database stats via `/health` endpoint

6. **CloudConvert Integration Issues**
   - Verify `CLOUDCONVERT_API_KEY` is set correctly in `.env`
   - Check CloudConvert account status and API limits
   - Ensure file format is supported by CloudConvert
   - Monitor conversion logs for specific error messages

### Logs and Debugging
- Application logs are displayed in the console
- Check `chroma_db/` directory for vector database files
- Monitor `uploads/` directory for temporary files
- Use `/health` endpoint to check system status
- Use `/tasks` endpoint to monitor processing tasks

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
- Monitor task status and logs for processing issues
- Open an issue on the repository

---

**Built with ❤️ using FastAPI, LlamaIndex, and Google Gemini**


