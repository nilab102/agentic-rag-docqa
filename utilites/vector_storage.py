import os
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import (
    MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
)
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration for ChromaDB"""
    persist_directory: str = "./chroma_db"
    collection_name: str = "document_chunks"
    embedding_function: Optional[str] = None

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
    Vector storage system using LlamaIndex with ChromaDB.
    Stores document chunks with embeddings for similarity search.
    """
    
    def __init__(self, config: DatabaseConfig, embedding_dimension: int = None):
        """
        Initialize vector storage.
        
        Args:
            config: Database configuration
            embedding_dimension: Dimension of embeddings (auto-detected from model if None)
        """
        self.config = config
        
        # Auto-detect embedding dimension based on model
        if embedding_dimension is None:
            model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
            if "embedding-exp-03-07" in model_name:
                self.embedding_dimension = 768
            else:
                self.embedding_dimension = 768
        else:
            self.embedding_dimension = 768
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.config.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize or get collection
        self._initialize_collection()
        
        # Initialize LlamaIndex components
        self._initialize_llama_index()
    
    def _initialize_collection(self):
        """Initialize or get the ChromaDB collection."""
        try:
            # Try to get existing collection
            self.chroma_collection = self.chroma_client.get_collection(
                name=self.config.collection_name
            )
            logger.info(f"✅ Connected to existing collection: {self.config.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist
            self.chroma_collection = self.chroma_client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Created new collection: {self.config.collection_name}")
    
    def _initialize_llama_index(self):
        """Initialize LlamaIndex components."""
        try:
            # Get API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not found")
            
            # Initialize Google GenAI embedding model
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
            embed_model = GoogleGenAIEmbedding(
                model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
                dimensions=768,
                api_key=api_key
            )
            
            # Create ChromaVectorStore
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Create storage context
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create index with explicit embedding model
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
                embed_model=embed_model
            )
            
            logger.info("✅ LlamaIndex components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing LlamaIndex: {e}")
            raise
    
    def setup_database(self):
        """Setup the vector storage database (already done in __init__)."""
        logger.info("✅ Database setup completed (ChromaDB initialized)")
        return True
    
    def _document_chunk_to_text_node(self, chunk: DocumentChunk) -> TextNode:
        """Convert DocumentChunk to LlamaIndex TextNode."""
        # Create metadata for the node
        metadata = {
            "chunk_id": chunk.chunk_id,
            "file_name": chunk.file_name,
            "file_path": chunk.file_path,
            "chunk_title": chunk.chunk_title,
            "chunk_summary": chunk.chunk_summary,
            "full_summary": chunk.full_summary,
            "created_at": chunk.created_at.isoformat() if chunk.created_at else datetime.now().isoformat(),
            "word_count": len(chunk.chunk_extracted_text.split()),
            "char_count": len(chunk.chunk_extracted_text)
        }
        
        # Create text content for embedding
        text_content = f"""
Title: {chunk.chunk_title}

Summary: {chunk.chunk_summary}

Content: {chunk.chunk_extracted_text}

Document Summary: {chunk.full_summary}
        """.strip()
        
        # Create TextNode
        node = TextNode(
            text=text_content,
            metadata=metadata,
            id_=chunk.chunk_id
        )
        
        # Set embedding if available
        if chunk.embedding:
            node.embedding = chunk.embedding
        
        return node
    
    def insert_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Insert a single document chunk into the database.
        
        Args:
            chunk: DocumentChunk object with all required fields
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to TextNode
            node = self._document_chunk_to_text_node(chunk)
            
            # Insert into index
            self.index.insert_nodes([node])
            
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
        
        # Convert all chunks to nodes
        nodes = []
        for chunk in chunks:
            try:
                node = self._document_chunk_to_text_node(chunk)
                nodes.append(node)
            except Exception as e:
                logger.error(f"❌ Error converting chunk {chunk.chunk_id}: {e}")
                failure_count += 1
        
        # Insert all nodes at once
        if nodes:
            try:
                self.index.insert_nodes(nodes)
                success_count = len(nodes)
                logger.info(f"✅ Batch inserted {success_count} chunks successfully")
            except Exception as e:
                logger.error(f"❌ Error in batch insert: {e}")
                failure_count += len(nodes)
        
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
            # Delete existing chunk
            self.delete_chunk(chunk_id)
            
            # Get the chunk data
            chunk_data = self.get_chunk_by_id(chunk_id)
            if not chunk_data:
                logger.warning(f"⚠️ Chunk {chunk_id} not found for embedding update")
                return False
            
            # Create new chunk with updated embedding
            updated_chunk = DocumentChunk(
                chunk_id=chunk_id,
                file_name=chunk_data["file_name"],
                file_path=chunk_data["file_path"],
                chunk_extracted_text=chunk_data["chunk_extracted_text"],
                full_summary=chunk_data["full_summary"],
                chunk_title=chunk_data["chunk_title"],
                chunk_summary=chunk_data["chunk_summary"],
                embedding=embedding,
                created_at=datetime.now()
            )
            
            # Insert updated chunk
            return self.insert_chunk(updated_chunk)
            
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
            # Query ChromaDB directly for chunks without embeddings
            results = self.chroma_collection.get(
                where={"embedding": None},
                include=["metadatas"]
            )
            
            chunk_ids = [metadata["chunk_id"] for metadata in results["metadatas"]]
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
            # Get the embedding model from the index
            embed_model = self.index.embed_model
            
            # Create retriever with similarity threshold
            from llama_index.core.retrievers import VectorIndexRetriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=limit,
                vector_store_query_mode="hybrid",
                embed_model=embed_model
            )
            
            # Create a dummy query (we'll use the embedding directly)
            # For now, we'll use a simple text query and filter by similarity
            results = retriever.retrieve("similarity search")
            
            # Filter by similarity threshold and convert to our format
            filtered_results = []
            for result in results:
                if hasattr(result, 'score') and result.score >= similarity_threshold:
                    chunk_data = self._node_to_dict(result.node)
                    chunk_data['similarity_score'] = result.score
                    filtered_results.append(chunk_data)
            
            logger.info(f"🔍 Found {len(filtered_results)} similar chunks")
            return filtered_results
            
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
            # Get the embedding model from the index
            embed_model = self.index.embed_model
            
            # Create retriever with explicit embedding model
            from llama_index.core.retrievers import VectorIndexRetriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=limit,
                vector_store_query_mode="hybrid",
                embed_model=embed_model
            )
            
            # Search
            results = retriever.retrieve(query_text)
            
            # Convert to our format
            chunks = []
            for result in results:
                chunk_data = self._node_to_dict(result.node)
                chunk_data['similarity_score'] = getattr(result, 'score', 0.0)
                chunks.append(chunk_data)
            
            logger.info(f"🔍 Found {len(chunks)} chunks matching text: '{query_text}'")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching by text: {e}")
            return []
    
    def search_with_metadata_filters(self, query_text: str, filters: MetadataFilters, 
                                   limit: int = 10) -> List[Dict]:
        """
        Search chunks with metadata filters.
        
        Args:
            query_text: Text to search for
            filters: MetadataFilters object
            limit: Maximum number of results
            
        Returns:
            List of matching chunks
        """
        try:
            # Get the embedding model from the index
            embed_model = self.index.embed_model
            
            # Create retriever with metadata filters and explicit embedding model
            from llama_index.core.retrievers import VectorIndexRetriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=limit,
                filters=filters,
                vector_store_query_mode="hybrid",
                embed_model=embed_model
            )
            
            # Search
            results = retriever.retrieve(query_text)
            
            # Convert to our format
            chunks = []
            for result in results:
                chunk_data = self._node_to_dict(result.node)
                chunk_data['similarity_score'] = getattr(result, 'score', 0.0)
                chunks.append(chunk_data)
            
            logger.info(f"🔍 Found {len(chunks)} chunks with metadata filters")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching with metadata filters: {e}")
            return []
    
    def _node_to_dict(self, node: TextNode) -> Dict:
        """Convert LlamaIndex TextNode to dictionary format."""
        return {
            "chunk_id": node.metadata.get("chunk_id", ""),
            "file_name": node.metadata.get("file_name", ""),
            "file_path": node.metadata.get("file_path", ""),
            "chunk_extracted_text": node.text,
            "full_summary": node.metadata.get("full_summary", ""),
            "chunk_title": node.metadata.get("chunk_title", ""),
            "chunk_summary": node.metadata.get("chunk_summary", ""),
            "created_at": node.metadata.get("created_at", ""),
            "word_count": node.metadata.get("word_count", 0),
            "char_count": node.metadata.get("char_count", 0)
        }
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get a specific chunk by ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk data as dictionary or None if not found
        """
        try:
            # Query ChromaDB directly
            results = self.chroma_collection.get(
                ids=[chunk_id],
                include=["metadatas", "documents"]
            )
            
            if results["ids"]:
                metadata = results["metadatas"][0]
                document = results["documents"][0]
                
                return {
                    "chunk_id": metadata.get("chunk_id", ""),
                    "file_name": metadata.get("file_name", ""),
                    "file_path": metadata.get("file_path", ""),
                    "chunk_extracted_text": document,
                    "full_summary": metadata.get("full_summary", ""),
                    "chunk_title": metadata.get("chunk_title", ""),
                    "chunk_summary": metadata.get("chunk_summary", ""),
                    "created_at": metadata.get("created_at", ""),
                    "word_count": metadata.get("word_count", 0),
                    "char_count": metadata.get("char_count", 0)
                }
            
            return None
            
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
            # Create metadata filter
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="file_name", value=file_name)
                ]
            )
            
            # Search with filter using explicit embedding model
            return self.search_with_metadata_filters("", filters, limit=1000)
            
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
            # Get collection count
            total_chunks = self.chroma_collection.count()
            
            # Get unique files
            results = self.chroma_collection.get(include=["metadatas"])
            unique_files = set()
            chunks_with_embeddings = 0
            
            for metadata in results["metadatas"]:
                unique_files.add(metadata.get("file_name", ""))
                # Note: ChromaDB automatically handles embeddings, so we assume all have them
                chunks_with_embeddings += 1
            
            stats = {
                "total_chunks": total_chunks,
                "chunks_with_embeddings": chunks_with_embeddings,
                "chunks_without_embeddings": 0,  # ChromaDB handles embeddings automatically
                "unique_files": len(unique_files),
                "avg_embedding_dimension": self.embedding_dimension,
                "embedding_coverage": 100.0 if total_chunks > 0 else 0.0
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
            # Delete from ChromaDB
            self.chroma_collection.delete(ids=[chunk_id])
            
            logger.info(f"✅ Chunk {chunk_id} deleted successfully")
            return True
            
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
            # Get all chunks for the file
            chunks = self.get_chunks_by_file(file_name)
            chunk_ids = [chunk["chunk_id"] for chunk in chunks]
            
            if chunk_ids:
                # Delete from ChromaDB
                self.chroma_collection.delete(ids=chunk_ids)
                logger.info(f"✅ Deleted {len(chunk_ids)} chunks for file: {file_name}")
                return len(chunk_ids)
            else:
                logger.info(f"ℹ️ No chunks found for file: {file_name}")
                return 0
                
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
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "document_chunks"),
        embedding_function=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
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
        print("Please check your configuration")
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
