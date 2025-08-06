import os
import asyncio
import logging
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our utilities
from utilites.pipline_upload import DocumentProcessingPipeline, PipelineConfig, ProcessingResult
from utilites.query import AdvancedQueryEngine, QueryConfig, QueryResult
from utilites.vector_storage import VectorStorage, DatabaseConfig, create_database_config_from_env

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG Document Q&A API",
    description="Upload documents and query them using advanced RAG capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
vector_storage: Optional[VectorStorage] = None
query_engine: Optional[AdvancedQueryEngine] = None
processing_pipeline: Optional[DocumentProcessingPipeline] = None

# Pydantic models for API requests/responses
class UploadResponse(BaseModel):
    success: bool
    filename: str
    message: str
    processing_details: Optional[Dict[str, Any]] = None
    timestamp: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query")
    limit: int = Field(10, description="Maximum number of results")
    use_rag: bool = Field(True, description="Whether to use RAG for response generation")
    file_filter: Optional[str] = Field(None, description="Filter results by specific filename")

class QueryResponse(BaseModel):
    success: bool
    query: str
    results_by_file: Dict[str, List[Dict[str, Any]]]
    total_files: int
    total_results: int
    rag_response: Optional[str] = None
    processing_time: float
    timestamp: str

class FileInfo(BaseModel):
    filename: str
    chunks_count: int
    uploaded_at: str

class FilesResponse(BaseModel):
    success: bool
    files: List[FileInfo]
    total_files: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    total_files: int
    total_chunks: int
    timestamp: str

async def initialize_services():
    """Initialize vector storage, query engine, and processing pipeline."""
    global vector_storage, query_engine, processing_pipeline
    
    try:
        # Initialize database configuration
        db_config = create_database_config_from_env()
        
        # Initialize vector storage
        vector_storage = VectorStorage(db_config)
        logger.info("✅ Vector storage initialized")
        
        # Initialize query engine
        query_config = QueryConfig()
        query_engine = AdvancedQueryEngine(vector_storage, query_config)
        logger.info("✅ Query engine initialized")
        
        # Initialize processing pipeline with custom configuration
        # Parse one page at a time, then chunk 3-5 pages after merging
        pipeline_config = PipelineConfig(
            db_config=db_config,
            # Parser settings - Parse one page at a time
            max_pages_per_chunk=1,
            boundary_sentences=2,
            boundary_table_rows=2,
            # Chunker settings - After merging, chunk 3-5 pages
            target_pages_per_chunk=4,  # Target middle of 3-5 range
            overlap_pages=1,
            max_pages_per_chunk_chunker=5,  # Max 5 pages
            min_pages_per_chunk=3,  # Min 3 pages
            respect_boundaries=True
        )
        processing_pipeline = DocumentProcessingPipeline(pipeline_config)
        logger.info("✅ Processing pipeline initialized with custom configuration")
        logger.info(f"   - Parse: 1 page at a time")
        logger.info(f"   - Chunk: 3-5 pages after merging")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await initialize_services()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Agentic RAG Document Q&A API",
        "version": "1.0.0",
        "configuration": {
            "parser": "1 page at a time",
            "chunker": "3-5 pages after merging"
        },
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "files": "/files",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        if not vector_storage:
            raise HTTPException(status_code=503, detail="Services not initialized")
        
        # Get database stats
        stats = vector_storage.get_database_stats()
        
        return HealthResponse(
            status="healthy",
            database_connected=True,
            total_files=stats.get("total_files", 0),
            total_chunks=stats.get("total_chunks", 0),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    max_pages_per_chunk: int = Form(1),  # Default to 1 page parsing
    target_pages_per_chunk: int = Form(4),  # Default to 4 pages chunking
    overlap_pages: int = Form(1),
    min_pages_per_chunk: int = Form(3),  # Default to 3 pages minimum
    max_pages_per_chunk_chunker: int = Form(5)  # Default to 5 pages maximum
):
    """
    Upload and process a document.
    
    Configuration:
    - Parse: 1 page at a time (max_pages_per_chunk=1)
    - Chunk: 3-5 pages after merging (min=3, max=5, target=4)
    
    Supported file types: PDF, DOCX, DOC, TXT, MD, RTF
    """
    try:
        if not processing_pipeline:
            raise HTTPException(status_code=503, detail="Processing pipeline not initialized")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.rtf'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Copy uploaded file to temporary location
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process the document with custom configuration
            logger.info(f"Processing document: {file.filename}")
            logger.info(f"Configuration: Parse 1 page, Chunk {min_pages_per_chunk}-{max_pages_per_chunk_chunker} pages")
            
            result: ProcessingResult = await processing_pipeline.process_document(temp_file_path)
            
            # Prepare response
            if result.success:
                return UploadResponse(
                    success=True,
                    filename=file.filename,
                    message=f"Document processed successfully. {result.chunks_processed} chunks created (3-5 pages each).",
                    processing_details={
                        "total_chunks": result.total_chunks,
                        "chunks_processed": result.chunks_processed,
                        "chunks_failed": result.chunks_failed,
                        "processing_time": result.processing_time,
                        "configuration": {
                            "parse_pages_per_chunk": 1,
                            "chunk_min_pages": min_pages_per_chunk,
                            "chunk_max_pages": max_pages_per_chunk_chunker,
                            "chunk_target_pages": target_pages_per_chunk
                        }
                    },
                    timestamp=datetime.now().isoformat()
                )
            else:
                return UploadResponse(
                    success=False,
                    filename=file.filename,
                    message=f"Document processing failed: {result.error_message}",
                    timestamp=datetime.now().isoformat()
                )
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using semantic search and RAG.
    Results are organized by filename for easy identification.
    """
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Query engine not initialized")
        
        start_time = datetime.now()
        
        # Perform query based on parameters
        if request.file_filter:
            # Search in specific file
            query_result = query_engine.search_by_file(
                request.query, 
                request.file_filter, 
                request.limit
            )
        else:
            # General search
            if request.use_rag:
                query_result = query_engine.rag_query(request.query, request.limit)
            else:
                query_result = query_engine.search_semantic(request.query, request.limit)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Group results by filename
        results_by_file = query_engine.get_results_by_file(query_result)
        
        return QueryResponse(
            success=True,
            query=request.query,
            results_by_file=results_by_file,
            total_files=len(results_by_file),
            total_results=query_result.total_results,
            rag_response=query_result.rag_response,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/files", response_model=FilesResponse)
async def list_files():
    """
    List all uploaded files with their metadata.
    """
    try:
        if not vector_storage:
            raise HTTPException(status_code=503, detail="Vector storage not initialized")
        
        # Get all files from database
        files = query_engine.list_files()
        
        # Get file information
        file_infos = []
        for filename in files:
            chunks = vector_storage.get_chunks_by_file(filename)
            file_infos.append(FileInfo(
                filename=filename,
                chunks_count=len(chunks),
                uploaded_at=chunks[0].get("created_at", "Unknown") if chunks else "Unknown"
            ))
        
        return FilesResponse(
            success=True,
            files=file_infos,
            total_files=len(file_infos),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """
    Delete a specific file and all its chunks from the database.
    """
    try:
        if not vector_storage:
            raise HTTPException(status_code=503, detail="Vector storage not initialized")
        
        # Delete all chunks for the file
        deleted_count = vector_storage.delete_file_chunks(filename)
        
        if deleted_count > 0:
            return {
                "success": True,
                "message": f"Deleted {deleted_count} chunks for file: {filename}",
                "deleted_chunks": deleted_count,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 