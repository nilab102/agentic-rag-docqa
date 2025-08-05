import os
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str

@dataclass
class DocumentChunk:
    """Represents a document chunk with all required fields"""
    chunk_id: str
    file_name: str
    file_path: str
    chunk_extracted_text: str
    full_summary: str
    chunk_title: str
    chunk_summary: str
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None

class VectorStorage:
    """
    Vector storage system using PostgreSQL with pgvector extension.
    Stores document chunks with embeddings for similarity search.
    """
    
    def __init__(self, config: DatabaseConfig, embedding_dimension: int = 1536):
        """
        Initialize vector storage.
        
        Args:
            config: Database configuration
            embedding_dimension: Dimension of embeddings (default 1536 for OpenAI)
        """
        self.config = config
        self.embedding_dimension = embedding_dimension
        
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist."""
        try:
            # Connect to default postgres database to create our database
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database="postgres",  # Connect to default postgres database
                user=self.config.username,
                password=self.config.password
            )
            conn.autocommit = True  # Required for creating databases
            cursor = conn.cursor()
            
            # Check if database exists (handle quoted names)
            db_name = self.config.database.strip('"')  # Remove quotes if present
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cursor.fetchone()
            
            if not exists:
                # Quote the database name to handle special characters like hyphens
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"✅ Created database: {db_name}")
            else:
                logger.info(f"✅ Database {db_name} already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating database: {e}")
            return False
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            # Handle database names with special characters
            db_name = self.config.database.strip('"')  # Remove quotes if present
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=db_name,
                user=self.config.username,
                password=self.config.password
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def setup_database(self):
        """Create the vector storage table and indexes."""
        
        # First, ensure the database exists
        db_name = self.config.database.strip('"')  # Remove quotes if present
        logger.info(f"🔍 Checking if database '{db_name}' exists...")
        if not self.create_database_if_not_exists():
            raise Exception(f"Failed to create database '{db_name}'")
        
        # Create pgvector extension
        create_extension_sql = """
        CREATE EXTENSION IF NOT EXISTS vector;
        """
        
        # Create the main table with all required columns
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(100) UNIQUE NOT NULL,
            file_name VARCHAR(500) NOT NULL,
            file_path TEXT NOT NULL,
            chunk_extracted_text TEXT NOT NULL,
            full_summary TEXT NOT NULL,
            chunk_title VARCHAR(1000) NOT NULL,
            chunk_summary TEXT NOT NULL,
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Indexes for better performance
            CONSTRAINT idx_chunk_id UNIQUE (chunk_id)
        );
        """
        
        # Create indexes
        create_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_file_name ON document_chunks(file_name);",
            "CREATE INDEX IF NOT EXISTS idx_file_path ON document_chunks(file_path);",
            "CREATE INDEX IF NOT EXISTS idx_created_at ON document_chunks(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_chunk_title ON document_chunks(chunk_title);",
        ]
        
        # Create vector index for similarity search
        create_vector_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_embedding_cosine 
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Create extension
                cursor.execute(create_extension_sql)
                logger.info("✅ pgvector extension created/verified")
                
                # Create table
                cursor.execute(create_table_sql)
                logger.info("✅ Document chunks table created")
                
                # Create basic indexes
                for index_sql in create_indexes_sql:
                    cursor.execute(index_sql)
                logger.info("✅ Basic indexes created")
                
                # Create vector index
                cursor.execute(create_vector_index_sql)
                logger.info("✅ Vector index created")
                
                conn.commit()
                logger.info("✅ Database setup completed successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"❌ Error setting up database: {e}")
                raise
    
    def insert_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Insert a single document chunk into the database.
        
        Args:
            chunk: DocumentChunk object with all required fields
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                insert_sql = """
                INSERT INTO document_chunks (
                    chunk_id, file_name, file_path, chunk_extracted_text,
                    full_summary, chunk_title, chunk_summary, embedding, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    file_path = EXCLUDED.file_path,
                    chunk_extracted_text = EXCLUDED.chunk_extracted_text,
                    full_summary = EXCLUDED.full_summary,
                    chunk_title = EXCLUDED.chunk_title,
                    chunk_summary = EXCLUDED.chunk_summary,
                    embedding = EXCLUDED.embedding,
                    created_at = EXCLUDED.created_at;
                """
                
                cursor.execute(insert_sql, (
                    chunk.chunk_id,
                    chunk.file_name,
                    chunk.file_path,
                    chunk.chunk_extracted_text,
                    chunk.full_summary,
                    chunk.chunk_title,
                    chunk.chunk_summary,
                    chunk.embedding,
                    chunk.created_at or datetime.now()
                ))
                
                conn.commit()
                logger.info(f"✅ Chunk {chunk.chunk_id} inserted successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error inserting chunk {chunk.chunk_id}: {e}")
            return False
    
    def insert_chunks_batch(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """
        Insert multiple document chunks in batch.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dict with success and failure counts
        """
        success_count = 0
        failure_count = 0
        
        for chunk in chunks:
            if self.insert_chunk(chunk):
                success_count += 1
            else:
                failure_count += 1
        
        logger.info(f"✅ Batch insert completed: {success_count} successful, {failure_count} failed")
        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(chunks)
        }
    
    def update_embedding(self, chunk_id: str, embedding: List[float]) -> bool:
        """
        Update embedding for a specific chunk.
        
        Args:
            chunk_id: ID of the chunk to update
            embedding: New embedding vector
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                update_sql = """
                UPDATE document_chunks 
                SET embedding = %s 
                WHERE chunk_id = %s
                """
                
                cursor.execute(update_sql, (embedding, chunk_id))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Embedding updated for chunk {chunk_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Chunk {chunk_id} not found for embedding update")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error updating embedding for chunk {chunk_id}: {e}")
            return False
    
    def get_chunks_without_embeddings(self) -> List[str]:
        """
        Get list of chunk IDs that don't have embeddings.
        
        Returns:
            List of chunk IDs
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT chunk_id FROM document_chunks WHERE embedding IS NULL")
                chunk_ids = [row[0] for row in cursor.fetchall()]
                
                logger.info(f"📊 Found {len(chunk_ids)} chunks without embeddings")
                return chunk_ids
                
        except Exception as e:
            logger.error(f"❌ Error getting chunks without embeddings: {e}")
            return []
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 10, 
                            similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                search_sql = """
                SELECT 
                    chunk_id,
                    file_name,
                    file_path,
                    chunk_extracted_text,
                    full_summary,
                    chunk_title,
                    chunk_summary,
                    1 - (embedding <=> %s) as similarity_score
                FROM document_chunks 
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> %s) >= %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """
                
                cursor.execute(search_sql, (
                    query_embedding, 
                    query_embedding, 
                    similarity_threshold,
                    query_embedding, 
                    limit
                ))
                
                results = cursor.fetchall()
                chunks = [dict(row) for row in results]
                
                logger.info(f"🔍 Found {len(chunks)} similar chunks")
                return chunks
                
        except Exception as e:
            logger.error(f"❌ Error searching similar chunks: {e}")
            return []
    
    def search_by_text(self, query_text: str, limit: int = 10) -> List[Dict]:
        """
        Search chunks by text content (full-text search).
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching chunks
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                search_sql = """
                SELECT 
                    chunk_id,
                    file_name,
                    file_path,
                    chunk_extracted_text,
                    full_summary,
                    chunk_title,
                    chunk_summary,
                    created_at
                FROM document_chunks 
                WHERE 
                    chunk_extracted_text ILIKE %s OR
                    chunk_title ILIKE %s OR
                    chunk_summary ILIKE %s OR
                    full_summary ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
                """
                
                search_pattern = f"%{query_text}%"
                cursor.execute(search_sql, (
                    search_pattern, search_pattern, search_pattern, search_pattern, limit
                ))
                
                results = cursor.fetchall()
                chunks = [dict(row) for row in results]
                
                logger.info(f"🔍 Found {len(chunks)} chunks matching text: '{query_text}'")
                return chunks
                
        except Exception as e:
            logger.error(f"❌ Error searching by text: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk data as dictionary or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT * FROM document_chunks WHERE chunk_id = %s
                """, (chunk_id,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"❌ Error getting chunk {chunk_id}: {e}")
            return None
    
    def get_chunks_by_file(self, file_name: str) -> List[Dict]:
        """
        Get all chunks for a specific file.
        
        Args:
            file_name: Name of the file
            
        Returns:
            List of chunks for the file
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT * FROM document_chunks 
                    WHERE file_name = %s 
                    ORDER BY created_at
                """, (file_name,))
                
                results = cursor.fetchall()
                chunks = [dict(row) for row in results]
                
                logger.info(f"📄 Found {len(chunks)} chunks for file: {file_name}")
                return chunks
                
        except Exception as e:
            logger.error(f"❌ Error getting chunks for file {file_name}: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total chunks
                cursor.execute("SELECT COUNT(*) FROM document_chunks")
                total_chunks = cursor.fetchone()[0]
                
                # Chunks with embeddings
                cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL")
                chunks_with_embeddings = cursor.fetchone()[0]
                
                # Unique files
                cursor.execute("SELECT COUNT(DISTINCT file_name) FROM document_chunks")
                unique_files = cursor.fetchone()[0]
                
                # Average embedding dimension (pgvector doesn't support array_length)
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM document_chunks 
                    WHERE embedding IS NOT NULL
                """)
                chunks_with_embeddings_count = cursor.fetchone()[0]
                
                # For pgvector, we know the dimension is fixed (1536)
                avg_dimension = 1536 if chunks_with_embeddings_count > 0 else 0
                
                stats = {
                    "total_chunks": total_chunks,
                    "chunks_with_embeddings": chunks_with_embeddings,
                    "chunks_without_embeddings": total_chunks - chunks_with_embeddings,
                    "unique_files": unique_files,
                    "avg_embedding_dimension": avg_dimension,
                    "embedding_coverage": (chunks_with_embeddings / total_chunks * 100) if total_chunks > 0 else 0
                }
                
                logger.info(f"📊 Database stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a specific chunk.
        
        Args:
            chunk_id: ID of the chunk to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM document_chunks WHERE chunk_id = %s", (chunk_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Chunk {chunk_id} deleted successfully")
                    return True
                else:
                    logger.warning(f"⚠️ Chunk {chunk_id} not found for deletion")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error deleting chunk {chunk_id}: {e}")
            return False
    
    def delete_file_chunks(self, file_name: str) -> int:
        """
        Delete all chunks for a specific file.
        
        Args:
            file_name: Name of the file
            
        Returns:
            int: Number of chunks deleted
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM document_chunks WHERE file_name = %s", (file_name,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"✅ Deleted {deleted_count} chunks for file: {file_name}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"❌ Error deleting chunks for file {file_name}: {e}")
            return 0

# Utility functions for working with the vector storage

def create_database_config_from_env() -> DatabaseConfig:
    """
    Create database configuration from environment variables.
    
    Returns:
        DatabaseConfig object
    """
    return DatabaseConfig(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "vector_storage_db"),
        username=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "")
    )

def create_chunk_from_data(file_name: str, file_path: str, chunk_text: str, 
                          full_summary: str, chunk_title: str, chunk_summary: str,
                          chunk_id: Optional[str] = None) -> DocumentChunk:
    """
    Create a DocumentChunk object from raw data.
    
    Args:
        file_name: Name of the file
        file_path: Path to the file
        chunk_text: Extracted text from the chunk
        full_summary: Summary of the entire document
        chunk_title: Title of the chunk
        chunk_summary: Summary of the chunk
        chunk_id: Optional chunk ID (will generate if not provided)
        
    Returns:
        DocumentChunk object
    """
    if chunk_id is None:
        chunk_id = str(uuid.uuid4())
    
    return DocumentChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        file_path=file_path,
        chunk_extracted_text=chunk_text,
        full_summary=full_summary,
        chunk_title=chunk_title,
        chunk_summary=chunk_summary
    )

# Example usage and testing functions

def example_usage():
    """Example of how to use the VectorStorage class."""
    
    # Create database config
    config = create_database_config_from_env()
    
    # Initialize vector storage
    vector_storage = VectorStorage(config)
    
    # Setup database (this will create the database if it doesn't exist)
    try:
        vector_storage.setup_database()
        print("✅ Database setup completed successfully")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("Please check your database credentials and ensure PostgreSQL is running")
        return
    
    # Create sample chunks
    chunks = [
        create_chunk_from_data(
            file_name="sample_doc.pdf",
            file_path="/path/to/sample_doc.pdf",
            chunk_text="This is the extracted text from the first chunk...",
            full_summary="This document discusses various topics including...",
            chunk_title="Introduction",
            chunk_summary="Introduction to the main topics"
        ),
        create_chunk_from_data(
            file_name="sample_doc.pdf",
            file_path="/path/to/sample_doc.pdf",
            chunk_text="This is the extracted text from the second chunk...",
            full_summary="This document discusses various topics including...",
            chunk_title="Methods",
            chunk_summary="Description of the methods used"
        )
    ]
    
    # Insert chunks
    result = vector_storage.insert_chunks_batch(chunks)
    print(f"Inserted {result['success_count']} chunks")
    
    # Get database stats
    stats = vector_storage.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Search by text
    results = vector_storage.search_by_text("methods")
    print(f"Found {len(results)} chunks containing 'methods'")

if __name__ == "__main__":
    example_usage()
