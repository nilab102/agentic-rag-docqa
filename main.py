import os
import asyncio
import logging
import tempfile
import shutil
import uuid
import threading
import time
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Import our utilities
from utilites.pipline_upload import DocumentProcessingPipeline, PipelineConfig, ProcessingResult
from utilites.query import AdvancedQueryEngine, QueryConfig, QueryResult
from utilites.vector_storage import VectorStorage, DatabaseConfig, create_database_config_from_env

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task status enumeration
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Task information storage
class TaskInfo(BaseModel):
    task_id: str
    filename: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0  # 0-100
    message: str = ""
    error_message: Optional[str] = None
    processing_details: Optional[Dict[str, Any]] = None

# Global task storage (in production, use Redis or database)
task_storage: Dict[str, TaskInfo] = {}

# Thread pool for background processing
thread_pool = ThreadPoolExecutor(max_workers=3)

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
    task_id: str
    filename: str
    message: str
    timestamp: str

class TaskStatusResponse(BaseModel):
    success: bool
    task_info: TaskInfo
    timestamp: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query")
    limit: int = Field(10, description="Maximum number of results")
    use_rag: bool = Field(True, description="Whether to use RAG for response generation")

class QueryResponse(BaseModel):
    success: bool
    file_names: List[str]
    rag_response: Optional[str] = None
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

class TaskListResponse(BaseModel):
    success: bool
    tasks: List[TaskInfo]
    total_tasks: int
    timestamp: str

def clean_filename(filename: str) -> str:
    """
    Remove UUID prefix from filename.
    Handles patterns like: 02c2a7f6-e24d-47eb-a94b-67060bb31261_originalname.pdf
    Returns: originalname.pdf
    """
    if not filename:
        return filename
    
    # Pattern to match UUID prefix: 8-4-4-4-12 hexadecimal characters followed by underscore
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_'
    
    # Remove UUID prefix if present
    cleaned = re.sub(uuid_pattern, '', filename, flags=re.IGNORECASE)
    
    return cleaned

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
        
        # Initialize processing pipeline with default configuration
        pipeline_config = PipelineConfig(db_config=db_config)
        processing_pipeline = DocumentProcessingPipeline(pipeline_config)
        logger.info("✅ Processing pipeline initialized with default configuration")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise

def simulate_progress_updates(task_id: str, start_progress: int = 10, end_progress: int = 99):
    """Simulate gradual progress updates every 2 seconds."""
    current_progress = start_progress
    
    while current_progress < end_progress and task_id in task_storage:
        # Check if task is still processing
        if task_storage[task_id].status != TaskStatus.PROCESSING:
            break
            
        # Update progress
        current_progress += 1
        if current_progress <= end_progress:
            task_storage[task_id].progress = current_progress
            
            # Update message based on progress
            if current_progress < 30:
                task_storage[task_id].message = "Initializing document processing..."
            elif current_progress < 50:
                task_storage[task_id].message = "Reading document content..."
            elif current_progress < 70:
                task_storage[task_id].message = "Extracting text and metadata..."
            elif current_progress < 90:
                task_storage[task_id].message = "Creating document chunks..."
            else:
                task_storage[task_id].message = "Storing document chunks in database..."
                
            logger.info(f"📊 Task {task_id}: Progress {current_progress}%")
        
        # Wait 2 seconds before next update
        time.sleep(2)

def process_document_background_sync(task_id: str, file_path: str, original_filename: str):
    """Background task to process uploaded document (runs in thread pool)."""
    global processing_pipeline, task_storage, vector_storage
    
    # Update task status to processing
    if task_id in task_storage:
        task_storage[task_id].status = TaskStatus.PROCESSING
        task_storage[task_id].started_at = datetime.now().isoformat()
        task_storage[task_id].progress = 10
        task_storage[task_id].message = "Starting document processing..."
        logger.info(f"📝 Starting background processing for task {task_id}: {original_filename}")
    
    # Start progress simulation in a separate thread
    progress_thread = threading.Thread(
        target=simulate_progress_updates,
        args=(task_id, 10, 99),
        daemon=True
    )
    progress_thread.start()
    
    try:
        # Process the document - handle both sync and async versions
        try:
            if asyncio.iscoroutinefunction(processing_pipeline.process_document):
                # If it's async, run it in an event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(processing_pipeline.process_document(file_path))
                loop.close()
            else:
                # If it's sync, call it directly
                result = processing_pipeline.process_document(file_path)
        except Exception as process_error:
            logger.error(f"Processing error: {process_error}")
            raise process_error
        
        # After processing, update the filename in the database if needed
        if result.success and vector_storage:
            try:
                # Get the temp filename that was stored
                temp_filename = os.path.basename(file_path)
                
                # Try to update filename if the method exists
                if hasattr(vector_storage, 'update_filename_for_chunks'):
                    vector_storage.update_filename_for_chunks(temp_filename, original_filename)
                    logger.info(f"📝 Updated filename from {temp_filename} to {original_filename}")
                else:
                    logger.info(f"📝 VectorStorage doesn't support filename updates. File stored as: {temp_filename}")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not update filename in database: {e}")
        
        # Update final task status - this will stop the progress simulation
        if task_id in task_storage:
            if result.success:
                task_storage[task_id].status = TaskStatus.COMPLETED
                task_storage[task_id].progress = 100
                task_storage[task_id].message = f"Document processed successfully. {result.chunks_processed} chunks created."
                task_storage[task_id].processing_details = {
                    "total_chunks": result.total_chunks,
                    "chunks_processed": result.chunks_processed,
                    "chunks_failed": result.chunks_failed,
                    "processing_time": result.processing_time
                }
            else:
                task_storage[task_id].status = TaskStatus.FAILED
                task_storage[task_id].progress = 0
                task_storage[task_id].error_message = result.error_message
                task_storage[task_id].message = f"Processing failed: {result.error_message}"
            
            task_storage[task_id].completed_at = datetime.now().isoformat()
            
        logger.info(f"✅ Background processing completed for task {task_id}: {original_filename}")
        
    except Exception as e:
        logger.error(f"❌ Background processing failed for task {task_id}: {e}")
        if task_id in task_storage:
            task_storage[task_id].status = TaskStatus.FAILED
            task_storage[task_id].progress = 0
            task_storage[task_id].error_message = str(e)
            task_storage[task_id].message = f"Processing failed: {str(e)}"
            task_storage[task_id].completed_at = datetime.now().isoformat()
    
    finally:
        # Wait for progress thread to finish (it will stop when status changes)
        if progress_thread.is_alive():
            progress_thread.join(timeout=5)
            
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"🗑️ Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up temporary file {file_path}: {e}")

def start_background_processing(task_id: str, file_path: str, original_filename: str):
    """Start processing in thread pool to ensure it's truly non-blocking."""
    future = thread_pool.submit(process_document_background_sync, task_id, file_path, original_filename)
    return future

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await initialize_services()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML UI."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>Error</title></head>
                <body>
                    <h1>Error: index.html not found</h1>
                    <p>Please make sure index.html is in the same directory as main.py</p>
                </body>
            </html>
            """,
            status_code=404
        )

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint."""
    return {
        "message": "Agentic RAG Document Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "query": "/query",
            "files": "/files",
            "health": "/health",
            "tasks": "/tasks",
            "task_status": "/tasks/{task_id}"
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a document in background.
    Returns immediately with task_id to track processing status.
    
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
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create temporary file with preserved extension
        temp_dir = tempfile.gettempdir()
        # Create a unique filename while preserving the original name structure
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
        temp_filename = f"{task_id}_{safe_filename}"
        temp_file_path = os.path.join(temp_dir, temp_filename)
        
        # Save uploaded file to temporary location
        try:
            with open(temp_file_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            filename=file.filename,  # Store original filename
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
            message="Document uploaded, queued for processing"
        )
        
        # Store task info
        task_storage[task_id] = task_info
        
        # Start background processing in thread pool (truly non-blocking)
        start_background_processing(task_id, temp_file_path, file.filename)
        
        logger.info(f"📤 Document uploaded and queued: {file.filename} (Task ID: {task_id})")
        
        return UploadResponse(
            success=True,
            task_id=task_id,
            filename=file.filename,
            message="Document uploaded successfully. Processing started in background.",
            timestamp=datetime.now().isoformat()
        )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a specific task.
    """
    try:
        if task_id not in task_storage:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        task_info = task_storage[task_id]
        
        return TaskStatusResponse(
            success=True,
            task_info=task_info,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter tasks by status"),
    limit: int = Query(50, description="Maximum number of tasks to return")
):
    """
    List all tasks, optionally filtered by status.
    """
    try:
        tasks = list(task_storage.values())
        
        # Filter by status if provided
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        # Limit results
        tasks = tasks[:limit]
        
        return TaskListResponse(
            success=True,
            tasks=tasks,
            total_tasks=len(tasks),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    Delete a task from the task storage.
    """
    try:
        if task_id not in task_storage:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        task_info = task_storage.pop(task_id)
        
        return {
            "success": True,
            "message": f"Task deleted: {task_id}",
            "deleted_task": {
                "task_id": task_info.task_id,
                "filename": task_info.filename,
                "status": task_info.status
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using semantic search and RAG.
    Returns only file names and RAG response.
    """
    try:
        if not query_engine:
            raise HTTPException(status_code=503, detail="Query engine not initialized")
        
        # Perform query - removed file_filter logic
        if request.use_rag:
            query_result = query_engine.rag_query(request.query, request.limit)
        else:
            query_result = query_engine.search_semantic(request.query, request.limit)
        
        # Extract unique file names from results and clean them
        raw_file_names = list(set(result.get('file_name', '') for result in query_result.results))
        raw_file_names = [name for name in raw_file_names if name]  # Remove empty names
        
        # Clean the filenames by removing UUID prefixes
        clean_file_names = [clean_filename(filename) for filename in raw_file_names]
        # Remove duplicates that might result from cleaning
        clean_file_names = list(set(clean_file_names))
        # Remove any empty strings that might result from cleaning
        clean_file_names = [name for name in clean_file_names if name]
        
        logger.info(f"🔍 Query: '{request.query}' - Found files: {clean_file_names}")
        
        return QueryResponse(
            success=True,
            file_names=clean_file_names,
            rag_response=query_result.rag_response,
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
        
        # Get file information and clean filenames
        file_infos = []
        for filename in files:
            chunks = vector_storage.get_chunks_by_file(filename)
            # Clean the filename for display
            clean_name = clean_filename(filename)
            if clean_name:  # Only add if cleaning resulted in a valid name
                file_infos.append(FileInfo(
                    filename=clean_name,
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
    Note: This endpoint expects the clean filename (without UUID prefix)
    """
    try:
        if not vector_storage:
            raise HTTPException(status_code=503, detail="Vector storage not initialized")
        
        # Get all files to find the matching one (including UUID prefix)
        all_files = query_engine.list_files()
        matching_file = None
        
        # Look for a file whose cleaned name matches the requested filename
        for stored_filename in all_files:
            if clean_filename(stored_filename) == filename:
                matching_file = stored_filename
                break
        
        if not matching_file:
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Delete all chunks for the file using the actual stored filename
        deleted_count = vector_storage.delete_file_chunks(matching_file)
        
        if deleted_count > 0:
            return {
                "success": True,
                "message": f"Deleted {deleted_count} chunks for file: {filename}",
                "deleted_chunks": deleted_count,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"No chunks found for file: {filename}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete file {filename}: {e}")
        
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", 1090)),
        log_level="info")