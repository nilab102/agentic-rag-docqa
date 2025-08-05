import os
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

# Import all required components
from parser import SmartDocumentProcessor
from chunker import SmartTextChunker, Chunk
from vector_storage import VectorStorage, DatabaseConfig, DocumentChunk, create_database_config_from_env

# Load environment variables
load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the document processing pipeline."""
    # Parser settings
    max_pages_per_chunk: int = 3
    boundary_sentences: int = 2
    boundary_table_rows: int = 2
    
    # Chunker settings
    target_pages_per_chunk: int = 5
    overlap_pages: int = 1
    max_pages_per_chunk_chunker: int = 5
    min_pages_per_chunk: int = 1
    respect_boundaries: bool = True
    
    # Embedding settings
    embedding_model: str = "gemini-embedding-exp-03-07"
    embedding_dimension: int = 1536
    embedding_task: str = "RETRIEVAL_DOCUMENT"
    
    # Database settings
    db_config: Optional[DatabaseConfig] = None
    
    # Processing settings
    batch_size: int = 10
    delay_between_requests: float = 1.0
    max_retries: int = 3

@dataclass
class ProcessingResult:
    """Result of document processing."""
    file_path: str
    file_name: str
    total_chunks: int
    chunks_processed: int
    chunks_failed: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    chunk_details: List[Dict[str, Any]] = None

class GeminiTextProcessor:
    """Handles text processing using Gemini AI."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def generate_full_summary(self, extracted_text: str) -> str:
        """Generate a concise summary of the entire document."""
        prompt = f"""
        You are an expert document summarizer. Please create a concise, comprehensive summary of the following document.
        
        Requirements:
        - Keep the summary under 300 words
        - Focus on main topics, key findings, and important conclusions
        - Maintain the document's structure and flow
        - Use clear, professional language
        - Include any important data, statistics, or key points
        
        Document content:
        {extracted_text[:8000]}  # Limit to first 8000 chars to avoid token limits
        
        Please provide a well-structured summary:
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating full summary: {e}")
            return "Document summary could not be generated."
    
    async def generate_chunk_metadata(self, chunk_text: str, chunk_index: int, total_chunks: int) -> Tuple[str, str]:
        """Generate title and summary for a chunk."""
        prompt = f"""
        You are analyzing chunk {chunk_index + 1} of {total_chunks} from a document.
        
        Please provide:
        1. A concise title (max 100 characters) that captures the main topic of this chunk
        2. A brief summary (max 200 words) that explains the key content and findings
        
        Chunk content:
        {chunk_text[:4000]}  # Limit to avoid token limits
        
        Format your response as:
        TITLE: [Your title here]
        SUMMARY: [Your summary here]
        """
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            response_text = response.text.strip()
            
            # Parse the response
            title = ""
            summary = ""
            
            lines = response_text.split('\n')
            for line in lines:
                if line.startswith('TITLE:'):
                    title = line.replace('TITLE:', '').strip()
                elif line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
            
            # Fallback if parsing fails
            if not title or not summary:
                title = f"Chunk {chunk_index + 1}"
                summary = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            
            return title, summary
            
        except Exception as e:
            logger.error(f"Error generating chunk metadata: {e}")
            return f"Chunk {chunk_index + 1}", chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text

class GeminiEmbeddingGenerator:
    """Handles embedding generation using Gemini."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-embedding-exp-03-07"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
    
    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT", 
                               output_dimensionality: int = 1536) -> Optional[List[float]]:
        """Generate embedding for text."""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=task_type,
                output_dimensionality=output_dimensionality
            )
            
            await asyncio.sleep(0.5)  # Rate limiting
            return result['embedding']
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT",
                                      output_dimensionality: int = 1536) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        for i, text in enumerate(texts):
            logger.info(f"Generating embedding {i+1}/{len(texts)}")
            embedding = await self.generate_embedding(text, task_type, output_dimensionality)
            embeddings.append(embedding)
        
        return embeddings

class DocumentProcessingPipeline:
    """Main pipeline for processing documents from parsing to vector storage."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Get API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found")
        
        # Initialize components
        self.parser = SmartDocumentProcessor(
            api_key=self.api_key,
            max_pages_per_chunk=config.max_pages_per_chunk,
            boundary_sentences=config.boundary_sentences,
            boundary_table_rows=config.boundary_table_rows
        )
        
        self.chunker = SmartTextChunker(
            target_pages_per_chunk=config.target_pages_per_chunk,
            overlap_pages=config.overlap_pages,
            max_pages_per_chunk=config.max_pages_per_chunk_chunker,
            min_pages_per_chunk=config.min_pages_per_chunk,
            respect_boundaries=config.respect_boundaries
        )
        
        self.text_processor = GeminiTextProcessor(self.api_key)
        self.embedding_generator = GeminiEmbeddingGenerator(self.api_key, config.embedding_model)
        
        # Initialize vector storage
        if config.db_config:
            self.vector_storage = VectorStorage(config.db_config, config.embedding_dimension)
        else:
            db_config = create_database_config_from_env()
            self.vector_storage = VectorStorage(db_config, config.embedding_dimension)
    
    def setup_database(self):
        """Setup the vector storage database."""
        try:
            self.vector_storage.setup_database()
            logger.info("✅ Database setup completed")
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
    
    async def process_document(self, file_path: str) -> ProcessingResult:
        """
        Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ProcessingResult with processing details
        """
        start_time = datetime.now()
        file_name = Path(file_path).name
        
        logger.info(f"🚀 Starting processing for: {file_name}")
        
        try:
            # Step 1: Parse document
            logger.info("📄 Step 1: Parsing document...")
            extracted_text = self.parser.extract_text_simple(file_path)
            logger.info(f"✅ Document parsed: {len(extracted_text)} characters")
            
            # Step 2: Generate full document summary
            logger.info("📝 Step 2: Generating document summary...")
            full_summary = await self.text_processor.generate_full_summary(extracted_text)
            logger.info(f"✅ Document summary generated: {len(full_summary)} characters")
            
            # Step 3: Chunk the document
            logger.info("✂️ Step 3: Chunking document...")
            chunks = self.chunker.chunk_text(extracted_text)
            logger.info(f"✅ Document chunked into {len(chunks)} chunks")
            
            # Step 4: Process each chunk
            logger.info("🔄 Step 4: Processing chunks...")
            processed_chunks = []
            failed_chunks = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    
                    # Generate chunk metadata
                    chunk_title, chunk_summary = await self.text_processor.generate_chunk_metadata(
                        chunk.content, i, len(chunks)
                    )
                    
                    # Prepare text for embedding (full_summary + chunk_title + chunk_summary)
                    embedding_text = f"{full_summary}\n\nTitle: {chunk_title}\n\nSummary: {chunk_summary}"
                    
                    # Generate embedding
                    embedding = await self.embedding_generator.generate_embedding(
                        embedding_text,
                        self.config.embedding_task,
                        self.config.embedding_dimension
                    )
                    
                    if embedding is None:
                        logger.warning(f"⚠️ Failed to generate embedding for chunk {i+1}")
                        failed_chunks += 1
                        continue
                    
                    # Create DocumentChunk object
                    document_chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        file_name=file_name,
                        file_path=file_path,
                        chunk_extracted_text=chunk.content,
                        full_summary=full_summary,
                        chunk_title=chunk_title,
                        chunk_summary=chunk_summary,
                        embedding=embedding,
                        created_at=datetime.now()
                    )
                    
                    # Store in vector database
                    success = self.vector_storage.insert_chunk(document_chunk)
                    
                    if success:
                        processed_chunks.append({
                            'chunk_id': document_chunk.chunk_id,
                            'title': chunk_title,
                            'summary': chunk_summary,
                            'word_count': chunk.word_count,
                            'page_numbers': chunk.page_numbers
                        })
                        logger.info(f"✅ Chunk {i+1} processed and stored successfully")
                    else:
                        logger.error(f"❌ Failed to store chunk {i+1}")
                        failed_chunks += 1
                
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {i+1}: {e}")
                    failed_chunks += 1
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ProcessingResult(
                file_path=file_path,
                file_name=file_name,
                total_chunks=len(chunks),
                chunks_processed=len(processed_chunks),
                chunks_failed=failed_chunks,
                processing_time=processing_time,
                success=len(processed_chunks) > 0,
                chunk_details=processed_chunks
            )
            
            logger.info(f"🎉 Processing completed for {file_name}")
            logger.info(f"   ✅ Successfully processed: {len(processed_chunks)} chunks")
            logger.info(f"   ❌ Failed: {failed_chunks} chunks")
            logger.info(f"   ⏱️  Total time: {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Processing failed for {file_name}: {e}")
            
            return ProcessingResult(
                file_path=file_path,
                file_name=file_name,
                total_chunks=0,
                chunks_processed=0,
                chunks_failed=0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def process_documents_batch(self, file_paths: List[str]) -> List[ProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ProcessingResult objects
        """
        logger.info(f"🚀 Starting batch processing of {len(file_paths)} documents")
        
        results = []
        
        for i, file_path in enumerate(file_paths):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing document {i+1}/{len(file_paths)}: {Path(file_path).name}")
            logger.info(f"{'='*60}")
            
            result = await self.process_document(file_path)
            results.append(result)
            
            # Add delay between documents to avoid rate limiting
            if i < len(file_paths) - 1:
                await asyncio.sleep(self.config.delay_between_requests)
        
        # Print batch summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[ProcessingResult]):
        """Print a summary of batch processing results."""
        total_files = len(results)
        successful_files = sum(1 for r in results if r.success)
        failed_files = total_files - successful_files
        
        total_chunks = sum(r.total_chunks for r in results)
        total_processed = sum(r.chunks_processed for r in results)
        total_failed = sum(r.chunks_failed for r in results)
        total_time = sum(r.processing_time for r in results)
        
        print(f"\n{'='*80}")
        print("📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"📄 FILES:")
        print(f"   Total files: {total_files}")
        print(f"   ✅ Successful: {successful_files}")
        print(f"   ❌ Failed: {failed_files}")
        print()
        print(f"🧩 CHUNKS:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   ✅ Processed: {total_processed}")
        print(f"   ❌ Failed: {total_failed}")
        print()
        print(f"⏱️  Total processing time: {total_time:.2f} seconds")
        print(f"📈 Average time per file: {total_time/total_files:.2f} seconds")
        print(f"{'='*80}")
        
        # Print individual file results
        print(f"\n📋 DETAILED RESULTS:")
        for result in results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"   {status} | {result.file_name} | {result.chunks_processed}/{result.total_chunks} chunks")
            if result.error_message:
                print(f"      Error: {result.error_message}")

# Main function for easy usage
async def process_document_pipeline(
    file_path: str,
    max_pages_per_chunk: int = 3,
    boundary_sentences: int = 2,
    boundary_table_rows: int = 2,
    target_pages_per_chunk: int = 5,
    overlap_pages: int = 1,
    max_pages_per_chunk_chunker: int = 5,
    min_pages_per_chunk: int = 1,
    respect_boundaries: bool = True,
    embedding_model: str = "gemini-embedding-exp-03-07",
    embedding_dimension: int = 1536,
    embedding_task: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 10,
    delay_between_requests: float = 1.0,
    max_retries: int = 3,
    db_config: Optional[DatabaseConfig] = None
) -> ProcessingResult:
    """
    Main function to process a document through the complete pipeline.
    
    Args:
        file_path: Path to the document file
        max_pages_per_chunk: Parser setting for max pages per chunk
        boundary_sentences: Parser setting for boundary sentences
        boundary_table_rows: Parser setting for boundary table rows
        target_pages_per_chunk: Chunker setting for target pages per chunk
        overlap_pages: Chunker setting for overlap pages
        max_pages_per_chunk_chunker: Chunker setting for max pages per chunk
        min_pages_per_chunk: Chunker setting for min pages per chunk
        respect_boundaries: Chunker setting for respecting boundaries
        embedding_model: Embedding model name
        embedding_dimension: Embedding dimension
        embedding_task: Embedding task type
        batch_size: Batch size for processing
        delay_between_requests: Delay between API requests
        max_retries: Maximum retries for API calls
        db_config: Database configuration (optional)
        
    Returns:
        ProcessingResult with processing details
    """
    
    # Create configuration
    config = PipelineConfig(
        max_pages_per_chunk=max_pages_per_chunk,
        boundary_sentences=boundary_sentences,
        boundary_table_rows=boundary_table_rows,
        target_pages_per_chunk=target_pages_per_chunk,
        overlap_pages=overlap_pages,
        max_pages_per_chunk_chunker=max_pages_per_chunk_chunker,
        min_pages_per_chunk=min_pages_per_chunk,
        respect_boundaries=respect_boundaries,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_task=embedding_task,
        batch_size=batch_size,
        delay_between_requests=delay_between_requests,
        max_retries=max_retries,
        db_config=db_config
    )
    
    # Initialize pipeline
    pipeline = DocumentProcessingPipeline(config)
    
    # Setup database
    pipeline.setup_database()
    
    # Process document
    result = await pipeline.process_document(file_path)
    
    return result

# Simple endpoint function for easy calling
async def process_document_endpoint(
    file_path: str,
    parser_config: Dict[str, Any] = None,
    chunker_config: Dict[str, Any] = None,
    embedding_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Simple endpoint function to process a document with default or custom configurations.
    
    Args:
        file_path: Path to the document file
        parser_config: Optional parser configuration dict
        chunker_config: Optional chunker configuration dict  
        embedding_config: Optional embedding configuration dict
        
    Returns:
        Dictionary with processing results
    """
    
    # Default configurations
    default_parser_config = {
        "max_pages_per_chunk": 3,
        "boundary_sentences": 2,
        "boundary_table_rows": 2
    }
    
    default_chunker_config = {
        "target_pages_per_chunk": 5,
        "overlap_pages": 1,
        "max_pages_per_chunk_chunker": 5,
        "min_pages_per_chunk": 1,
        "respect_boundaries": True
    }
    
    default_embedding_config = {
        "embedding_model": "gemini-embedding-exp-03-07",
        "embedding_dimension": 1536,
        "embedding_task": "RETRIEVAL_DOCUMENT",
        "batch_size": 10,
        "delay_between_requests": 1.0,
        "max_retries": 3
    }
    
    # Merge with provided configurations
    if parser_config:
        default_parser_config.update(parser_config)
    if chunker_config:
        default_chunker_config.update(chunker_config)
    if embedding_config:
        default_embedding_config.update(embedding_config)
    
    try:
        # Process the document
        result = await process_document_pipeline(
            file_path=file_path,
            max_pages_per_chunk=default_parser_config["max_pages_per_chunk"],
            boundary_sentences=default_parser_config["boundary_sentences"],
            boundary_table_rows=default_parser_config["boundary_table_rows"],
            target_pages_per_chunk=default_chunker_config["target_pages_per_chunk"],
            overlap_pages=default_chunker_config["overlap_pages"],
            max_pages_per_chunk_chunker=default_chunker_config["max_pages_per_chunk_chunker"],
            min_pages_per_chunk=default_chunker_config["min_pages_per_chunk"],
            respect_boundaries=default_chunker_config["respect_boundaries"],
            embedding_model=default_embedding_config["embedding_model"],
            embedding_dimension=default_embedding_config["embedding_dimension"],
            embedding_task=default_embedding_config["embedding_task"],
            batch_size=default_embedding_config["batch_size"],
            delay_between_requests=default_embedding_config["delay_between_requests"],
            max_retries=default_embedding_config["max_retries"]
        )
        
        # Convert result to dictionary
        return {
            "success": result.success,
            "file_name": result.file_name,
            "file_path": result.file_path,
            "total_chunks": result.total_chunks,
            "chunks_processed": result.chunks_processed,
            "chunks_failed": result.chunks_failed,
            "processing_time": result.processing_time,
            "error_message": result.error_message,
            "chunk_details": result.chunk_details or [],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "file_name": Path(file_path).name if file_path else "Unknown",
            "file_path": file_path,
            "total_chunks": 0,
            "chunks_processed": 0,
            "chunks_failed": 0,
            "processing_time": 0.0,
            "error_message": str(e),
            "chunk_details": [],
            "timestamp": datetime.now().isoformat()
        }

# FastAPI integration (optional)
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from pydantic import BaseModel
    from typing import Optional as OptionalType
    import tempfile
    import shutil
    
    # Pydantic models for API
    class ProcessingRequest(BaseModel):
        parser_config: OptionalType[Dict[str, Any]] = None
        chunker_config: OptionalType[Dict[str, Any]] = None
        embedding_config: OptionalType[Dict[str, Any]] = None
    
    class ProcessingResponse(BaseModel):
        success: bool
        file_name: str
        total_chunks: int
        chunks_processed: int
        chunks_failed: int
        processing_time: float
        error_message: OptionalType[str] = None
        chunk_details: List[Dict[str, Any]] = []
        timestamp: str
    
    # Create FastAPI app
    app = FastAPI(
        title="Document Processing Pipeline API",
        description="API for processing documents through parsing, chunking, and vector storage",
        version="1.0.0"
    )
    
    @app.post("/process-document", response_model=ProcessingResponse)
    async def process_document_api(
        file: UploadFile = File(...),
        parser_config: OptionalType[Dict[str, Any]] = None,
        chunker_config: OptionalType[Dict[str, Any]] = None,
        embedding_config: OptionalType[Dict[str, Any]] = None
    ):
        """
        Process a document through the complete pipeline.
        
        Upload a document file and get back processing results with chunk details.
        """
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Create temporary file
        temp_file = None
        try:
            # Save uploaded file to temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            
            # Process the document
            result = await process_document_endpoint(
                file_path=temp_file.name,
                parser_config=parser_config,
                chunker_config=chunker_config,
                embedding_config=embedding_config
            )
            
            return ProcessingResponse(**result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Document Processing Pipeline API",
            "version": "1.0.0",
            "endpoints": {
                "process_document": "/process-document",
                "health": "/health"
            },
            "documentation": "/docs"
        }

except ImportError:
    # FastAPI not available, create a simple mock
    app = None
    logger.info("FastAPI not available - web endpoints disabled")

# Simple function for direct usage
def process_document_simple(file_path: str) -> Dict[str, Any]:
    """
    Simple synchronous wrapper for processing a document.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Run the async function
        result = asyncio.run(process_document_endpoint(file_path))
        return result
    except Exception as e:
        return {
            "success": False,
            "file_name": Path(file_path).name if file_path else "Unknown",
            "file_path": file_path,
            "total_chunks": 0,
            "chunks_processed": 0,
            "chunks_failed": 0,
            "processing_time": 0.0,
            "error_message": str(e),
            "chunk_details": [],
            "timestamp": datetime.now().isoformat()
        }

# Example usage function
async def example_usage():
    """Example of how to use the document processing pipeline."""
    
    # Example file path - update this to your actual file
    file_path = "/Users/nilab/Desktop/projects/agentic-rag-docqa/Saudi-Arabia-findings.pdf"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please update the file_path in example_usage() to point to your document.")
        return
    
    try:
        print("🚀 Starting document processing pipeline...")
        
        # Process the document
        result = await process_document_pipeline(
            file_path=file_path,
            max_pages_per_chunk=21,
            boundary_sentences=2,
            boundary_table_rows=2,
            target_pages_per_chunk=21,
            overlap_pages=1,
            max_pages_per_chunk_chunker=5,
            min_pages_per_chunk=1,
            respect_boundaries=True,
            embedding_model="gemini-embedding-exp-03-07",
            embedding_dimension=1536,
            embedding_task="RETRIEVAL_DOCUMENT",
            batch_size=10,
            delay_between_requests=1.0,
            max_retries=3
        )
        
        if result.success:
            print(f"✅ Processing completed successfully!")
            print(f"   File: {result.file_name}")
            print(f"   Chunks processed: {result.chunks_processed}/{result.total_chunks}")
            print(f"   Processing time: {result.processing_time:.2f} seconds")
        else:
            print(f"❌ Processing failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Error in example usage: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
