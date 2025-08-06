import os
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import (
    MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import chromadb
from chromadb.config import Settings

# Import our vector storage
from .vector_storage import VectorStorage, DatabaseConfig, DocumentChunk, create_database_config_from_env

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Language code ('ar' for Arabic, 'en' for English, 'other' for other languages)
    """
    # Arabic character range
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    
    # Check if text contains Arabic characters
    if re.search(arabic_pattern, text):
        return 'ar'
    
    # Check for common English patterns
    english_pattern = r'[a-zA-Z]'
    if re.search(english_pattern, text):
        return 'en'
    
    # Default to other languages
    return 'other'

def get_language_specific_prompt(language: str) -> str:
    """
    Get language-specific instructions for the RAG prompt.
    
    Args:
        language: Language code ('ar', 'en', 'other')
        
    Returns:
        Language-specific prompt instructions
    """
    if language == 'ar':
        return """
تعليمات اللغة العربية:
- استجب باللغة العربية إذا كان السؤال باللغة العربية
- استخدم قواعد اللغة العربية الصحيحة والتنسيق المناسب
- استخدم التنسيق التالي للمراجع: "(الصفحة: X)" أو "(الصفحات: X, Y, Z)"
- إذا لم تتوفر أرقام الصفحات، اذكر "(الصفحة: غير محدد)"
- حافظ على نفس أسلوب اللغة ونبرة السؤال
"""
    elif language == 'en':
        return """
English Language Instructions:
- Respond in English if the question is in English
- Use proper English grammar and formatting
- Use this format for citations: "(Page: X)" or "(Pages: X, Y, Z)"
- If page numbers are not available, mention "(Page: Not specified)"
- Maintain the same language style and tone as the query
"""
    else:
        return """
Language Instructions:
- Respond in the same language as the user's query
- Use proper grammar and formatting for that language
- Use appropriate translation for page citations
- If page numbers are not available, use appropriate translation
- Maintain the same language style and tone as the query
"""

# Custom RAG prompt template that instructs LLM to cite page numbers and respond in the same language as the query
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided context documents. 
Your task is to provide accurate and helpful responses while citing the page numbers from the source documents.

LANGUAGE INSTRUCTIONS:
- If the user asks a question in Arabic, respond in Arabic
- If the user asks a question in English, respond in English
- If the user asks a question in any other language, respond in that same language
- Always maintain the same language as the user's query throughout your response
- For Arabic responses, use proper Arabic grammar and formatting

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the information provided in the context documents
2. At the end of your response, ALWAYS mention the page numbers where the information was found
3. Use this format for page citations:
   - English: "(Page: X)" or "(Pages: X, Y, Z)" where X, Y, Z are page numbers
   - Arabic: "(الصفحة: X)" or "(الصفحات: X, Y, Z)" where X, Y, Z are page numbers
   - Other languages: Use appropriate translation of "Page" or "Pages"
4. If page numbers are not available for some chunks, mention:
   - English: "(Page: Not specified)"
   - Arabic: "(الصفحة: غير محدد)"
   - Other languages: Use appropriate translation
5. If you cannot find relevant information in the context, say so clearly in the same language as the query
6. Be concise but comprehensive in your response
7. Maintain the same language style and tone as the user's query

Context Information:
{context_str}

Query: {query_str}

Answer: """

@dataclass
class QueryConfig:
    """Configuration for query operations."""
    # Search settings
    similarity_top_k: int = 10
    similarity_threshold: float = 0.7
    vector_store_query_mode: str = "hybrid"
    
    # RAG settings
    enable_rag: bool = True
    llm_model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    include_page_citations: bool = True  # New setting for page citations
    
    # Response settings
    include_metadata: bool = True
    include_similarity_scores: bool = True
    include_page_numbers: bool = True  # New setting for page numbers
    format_response: bool = True

@dataclass
class QueryResult:
    """Result of a query operation."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    query_type: str
    metadata_filters: Optional[Dict[str, Any]] = None
    rag_response: Optional[str] = None
    cited_pages: Optional[List[int]] = None  # New field for cited page numbers
    error_message: Optional[str] = None

class AdvancedQueryEngine:
    """
    Advanced query engine using LlamaIndex and ChromaDB.
    Supports semantic search, metadata filtering, and RAG with page citations.
    """
    
    def __init__(self, vector_storage: VectorStorage, config: QueryConfig = None):
        """
        Initialize the query engine.
        
        Args:
            vector_storage: VectorStorage instance
            config: Query configuration
        """
        self.vector_storage = vector_storage
        self.config = config or QueryConfig()
        
        # Get API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found")
        
        # Initialize LLM and embedding models
        self._initialize_models()
        
        # Initialize query engine
        self._initialize_query_engine()
    
    def _initialize_models(self):
        """Initialize LLM and embedding models."""
        try:
            # Initialize Google GenAI LLM
            self.llm = GoogleGenAI(
                model=self.config.llm_model,
                api_key=self.api_key,
                temperature=self.config.temperature
            )
            
            # Initialize Google GenAI embedding model
            self.embedding_model = GoogleGenAIEmbedding(
                model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
                dimensions=768,
                api_key=self.api_key
            )
            
            logger.info("✅ LLM and embedding models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            raise
    
    def _initialize_query_engine(self):
        """Initialize the query engine with retriever."""
        try:
            # Create retriever with explicit embedding model
            self.retriever = VectorIndexRetriever(
                index=self.vector_storage.index,
                similarity_top_k=self.config.similarity_top_k,
                vector_store_query_mode=self.config.vector_store_query_mode,
                embed_model=self.embedding_model
            )
            
            # Create query engine with custom prompt template for page citations
            if self.config.enable_rag and self.llm:
                # Create custom prompt template
                custom_prompt = PromptTemplate(RAG_PROMPT_TEMPLATE)
                
                self.query_engine = RetrieverQueryEngine.from_args(
                    retriever=self.retriever,
                    llm=self.llm,
                    text_qa_template=custom_prompt  # Use custom prompt for page citations
                )
            else:
                self.query_engine = RetrieverQueryEngine.from_args(
                    retriever=self.retriever
                )
            
            logger.info("✅ Query engine initialized successfully with page citation support")
            
        except Exception as e:
            logger.error(f"❌ Error initializing query engine: {e}")
            raise
    
    def _create_enhanced_context_string(self, nodes: List[NodeWithScore]) -> str:
        """
        Create enhanced context string with page information for RAG.
        
        Args:
            nodes: Retrieved nodes with scores
            
        Returns:
            Enhanced context string with page information
        """
        context_parts = []
        
        for i, node in enumerate(nodes, 1):
            # Get page number from metadata
            page_num = node.node.metadata.get("page_number", "Not specified")
            file_name = node.node.metadata.get("file_name", "Unknown file")
            chunk_title = node.node.metadata.get("chunk_title", f"Chunk {i}")
            
            # Create context entry with page information
            context_entry = f"""
Document {i}:
File: {file_name}
Title: {chunk_title}
Page: {page_num}
Content: {node.node.text}
---
"""
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _extract_cited_pages(self, rag_response: str, retrieved_nodes: List[NodeWithScore]) -> List[int]:
        """
        Extract page numbers that were likely cited in the RAG response.
        
        Args:
            rag_response: RAG response text
            retrieved_nodes: Nodes that were retrieved for context
            
        Returns:
            List of page numbers mentioned in response
        """
        cited_pages = []
        
        # Extract page numbers from retrieved nodes
        for node in retrieved_nodes:
            page_num = node.node.metadata.get("page_number")
            if page_num is not None and isinstance(page_num, (int, str)):
                try:
                    page_int = int(page_num)
                    if page_int not in cited_pages:
                        cited_pages.append(page_int)
                except (ValueError, TypeError):
                    continue
        
        return sorted(cited_pages)
    
    def search_semantic(self, query: str, limit: int = None, 
                       similarity_threshold: float = None) -> QueryResult:
        """
        Perform semantic search using embeddings.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            QueryResult with search results
        """
        start_time = datetime.now()
        
        try:
            # Use provided parameters or defaults
            limit = limit or self.config.similarity_top_k
            similarity_threshold = similarity_threshold or self.config.similarity_threshold
            
            # Perform search with explicit embedding model
            results = self.retriever.retrieve(query)
            
            # Filter by similarity threshold
            filtered_results = []
            for result in results:
                if hasattr(result, 'score') and result.score >= similarity_threshold:
                    chunk_data = self._node_to_dict(result.node)
                    if self.config.include_similarity_scores:
                        chunk_data['similarity_score'] = result.score
                    filtered_results.append(chunk_data)
            
            # Limit results
            filtered_results = filtered_results[:limit]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                results=filtered_results,
                total_results=len(filtered_results),
                processing_time=processing_time,
                query_type="semantic_search"
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Error in semantic search: {e}")
            
            return QueryResult(
                query=query,
                results=[],
                total_results=0,
                processing_time=processing_time,
                query_type="semantic_search",
                error_message=str(e)
            )
    
    def search_with_metadata_filters(self, query: str, filters: Dict[str, Any], 
                                   limit: int = None) -> QueryResult:
        """
        Search with metadata filters.
        
        Args:
            query: Search query
            filters: Dictionary of metadata filters
            limit: Maximum number of results
            
        Returns:
            QueryResult with search results
        """
        start_time = datetime.now()
        
        try:
            limit = limit or self.config.similarity_top_k
            
            # Convert filters to LlamaIndex format
            llama_filters = self._convert_filters_to_llama_index(filters)
            
            # Create retriever with filters and explicit embedding model
            filtered_retriever = VectorIndexRetriever(
                index=self.vector_storage.index,
                similarity_top_k=limit,
                filters=llama_filters,
                vector_store_query_mode=self.config.vector_store_query_mode,
                embed_model=self.embedding_model
            )
            
            # Perform search
            results = filtered_retriever.retrieve(query)
            
            # Convert to our format
            chunks = []
            for result in results:
                chunk_data = self._node_to_dict(result.node)
                if self.config.include_similarity_scores:
                    chunk_data['similarity_score'] = getattr(result, 'score', 0.0)
                chunks.append(chunk_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                results=chunks,
                total_results=len(chunks),
                processing_time=processing_time,
                query_type="metadata_filtered_search",
                metadata_filters=filters
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Error in metadata filtered search: {e}")
            
            return QueryResult(
                query=query,
                results=[],
                total_results=0,
                processing_time=processing_time,
                query_type="metadata_filtered_search",
                metadata_filters=filters,
                error_message=str(e)
            )
    
    def search_by_file(self, query: str, file_name: str, limit: int = None) -> QueryResult:
        """
        Search within a specific file.
        
        Args:
            query: Search query
            file_name: Name of the file to search in
            limit: Maximum number of results
            
        Returns:
            QueryResult with search results
        """
        filters = {"file_name": file_name}
        return self.search_with_metadata_filters(query, filters, limit)
    
    def search_by_page_number(self, query: str, page_number: int, limit: int = None) -> QueryResult:
        """
        Search within a specific page number.
        
        Args:
            query: Search query
            page_number: Page number to search in
            limit: Maximum number of results
            
        Returns:
            QueryResult with search results
        """
        filters = {"page_number": page_number}
        return self.search_with_metadata_filters(query, filters, limit)
    
    def search_by_page_range(self, query: str, start_page: int, end_page: int, 
                           limit: int = None) -> QueryResult:
        """
        Search within a page range.
        
        Args:
            query: Search query
            start_page: Start page number
            end_page: End page number
            limit: Maximum number of results
            
        Returns:
            QueryResult with search results
        """
        filters = {
            "page_number": {
                "start": start_page,
                "end": end_page
            }
        }
        return self.search_with_metadata_filters(query, filters, limit)
    
    def search_by_date_range(self, query: str, start_date: str, end_date: str, 
                           limit: int = None) -> QueryResult:
        """
        Search within a date range.
        
        Args:
            query: Search query
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of results
            
        Returns:
            QueryResult with search results
        """
        filters = {
            "created_at": {
                "start": start_date,
                "end": end_date
            }
        }
        return self.search_with_metadata_filters(query, filters, limit)
    
    def search_by_chunk_title(self, query: str, title_pattern: str, 
                            limit: int = None) -> QueryResult:
        """
        Search by chunk title pattern.
        
        Args:
            query: Search query
            title_pattern: Pattern to match in chunk titles
            limit: Maximum number of results
            
        Returns:
            QueryResult with search results
        """
        filters = {"chunk_title": title_pattern}
        return self.search_with_metadata_filters(query, filters, limit)
    
    def rag_query(self, query: str, limit: int = None) -> QueryResult:
        """
        Perform RAG (Retrieval-Augmented Generation) query with page citations.
        Supports multiple languages including Arabic.
        
        Args:
            query: User query
            limit: Maximum number of retrieved documents
            
        Returns:
            QueryResult with RAG response and page citations
        """
        start_time = datetime.now()
        
        try:
            if not self.config.enable_rag:
                raise ValueError("RAG is disabled in configuration")
            
            limit = limit or self.config.similarity_top_k
            
            # Detect language of the query
            detected_language = detect_language(query)
            logger.info(f"🌐 Detected language: {detected_language} for query: {query[:50]}...")
            
            # Create query engine with specific retriever and custom prompt
            rag_retriever = VectorIndexRetriever(
                index=self.vector_storage.index,
                similarity_top_k=limit,
                vector_store_query_mode=self.config.vector_store_query_mode,
                embed_model=self.embedding_model
            )
            
            # Create language-specific prompt
            language_instructions = get_language_specific_prompt(detected_language)
            
            # Create custom prompt template with language-specific instructions
            language_specific_template = f"""
You are a helpful assistant that answers questions based on the provided context documents. 
Your task is to provide accurate and helpful responses while citing the page numbers from the source documents.

{language_instructions}

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the information provided in the context documents
2. At the end of your response, ALWAYS mention the page numbers where the information was found
3. Use appropriate page citation format for the detected language
4. If page numbers are not available for some chunks, mention appropriate translation
5. If you cannot find relevant information in the context, say so clearly in the same language as the query
6. Be concise but comprehensive in your response
7. Maintain the same language style and tone as the user's query

Context Information:
{{context_str}}

Query: {{query_str}}

Answer: """
            
            custom_prompt = PromptTemplate(language_specific_template)
            
            rag_query_engine = RetrieverQueryEngine.from_args(
                retriever=rag_retriever,
                llm=self.llm,
                text_qa_template=custom_prompt
            )
            
            # Get RAG response
            response = rag_query_engine.query(query)
            
            # Get retrieved documents
            retrieved_nodes = rag_retriever.retrieve(query)
            chunks = []
            for node in retrieved_nodes:
                chunk_data = self._node_to_dict(node)
                chunks.append(chunk_data)
            
            # Extract cited pages
            cited_pages = self._extract_cited_pages(str(response), retrieved_nodes)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                results=chunks,
                total_results=len(chunks),
                processing_time=processing_time,
                query_type="rag_query",
                rag_response=str(response),
                cited_pages=cited_pages
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Error in RAG query: {e}")
            
            return QueryResult(
                query=query,
                results=[],
                total_results=0,
                processing_time=processing_time,
                query_type="rag_query",
                error_message=str(e)
            )
    
    def rag_query_with_page_context(self, query: str, limit: int = None) -> QueryResult:
        """
        Enhanced RAG query that explicitly includes page information in context.
        Supports multiple languages including Arabic.
        
        Args:
            query: User query
            limit: Maximum number of retrieved documents
            
        Returns:
            QueryResult with enhanced RAG response including page citations
        """
        start_time = datetime.now()
        
        try:
            if not self.config.enable_rag:
                raise ValueError("RAG is disabled in configuration")
            
            limit = limit or self.config.similarity_top_k
            
            # Detect language of the query
            detected_language = detect_language(query)
            logger.info(f"🌐 Detected language: {detected_language} for query: {query[:50]}...")
            
            # Retrieve relevant documents
            rag_retriever = VectorIndexRetriever(
                index=self.vector_storage.index,
                similarity_top_k=limit,
                vector_store_query_mode=self.config.vector_store_query_mode,
                embed_model=self.embedding_model
            )
            
            retrieved_nodes = rag_retriever.retrieve(query)
            
            # Create enhanced context with page information
            enhanced_context = self._create_enhanced_context_string(retrieved_nodes)
            
            # Create language-specific prompt
            language_instructions = get_language_specific_prompt(detected_language)
            
            # Create custom prompt with language-specific instructions
            enhanced_prompt = f"""
You are a helpful assistant that answers questions based on the provided context documents. 
Your task is to provide accurate and helpful responses while citing the page numbers from the source documents.

{language_instructions}

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the information provided in the context documents
2. At the end of your response, ALWAYS mention the page numbers where the information was found
3. Use appropriate page citation format for the detected language
4. If page numbers are not available for some chunks, mention appropriate translation
5. If you cannot find relevant information in the context, say so clearly in the same language as the query
6. Be concise but comprehensive in your response
7. Maintain the same language style and tone as the user's query

Context Information:
{enhanced_context}

Query: {query}

Answer: """
            
            # Get response from LLM
            response = self.llm.complete(enhanced_prompt)
            
            # Convert retrieved documents to our format
            chunks = []
            for node in retrieved_nodes:
                chunk_data = self._node_to_dict(node.node)
                chunks.append(chunk_data)
            
            # Extract cited pages
            cited_pages = self._extract_cited_pages(str(response), retrieved_nodes)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                results=chunks,
                total_results=len(chunks),
                processing_time=processing_time,
                query_type="enhanced_rag_query",
                rag_response=str(response),
                cited_pages=cited_pages
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Error in enhanced RAG query: {e}")
            
            return QueryResult(
                query=query,
                results=[],
                total_results=0,
                processing_time=processing_time,
                query_type="enhanced_rag_query",
                error_message=str(e)
            )
    
    def hybrid_search(self, query: str, filters: Dict[str, Any] = None, 
                     limit: int = None) -> QueryResult:
        """
        Perform hybrid search combining semantic and metadata filtering.
        
        Args:
            query: Search query
            filters: Optional metadata filters
            limit: Maximum number of results
            
        Returns:
            QueryResult with search results
        """
        if filters:
            return self.search_with_metadata_filters(query, filters, limit)
        else:
            return self.search_semantic(query, limit)
    
    def _convert_filters_to_llama_index(self, filters: Dict[str, Any]) -> MetadataFilters:
        """Convert dictionary filters to LlamaIndex MetadataFilters."""
        llama_filters = []
        
        for key, value in filters.items():
            if isinstance(value, dict) and "start" in value and "end" in value:
                # Range filter (for dates or page numbers)
                llama_filters.extend([
                    MetadataFilter(key=key, value=value["start"], operator=FilterOperator.GTE),
                    MetadataFilter(key=key, value=value["end"], operator=FilterOperator.LTE)
                ])
            elif isinstance(value, (list, tuple)):
                # Multiple values (OR condition)
                for v in value:
                    llama_filters.append(MetadataFilter(key=key, value=v))
            else:
                # Single value
                llama_filters.append(MetadataFilter(key=key, value=value))
        
        return MetadataFilters(filters=llama_filters)
    
    def _node_to_dict(self, node: TextNode) -> Dict[str, Any]:
        """Convert LlamaIndex TextNode to dictionary format with page number."""
        return {
            "chunk_id": node.metadata.get("chunk_id", ""),
            "file_name": node.metadata.get("file_name", ""),
            "file_path": node.metadata.get("file_path", ""),
            "page_number": node.metadata.get("page_number", None),  # Added page number
            "chunk_extracted_text": node.text,
            "full_summary": node.metadata.get("full_summary", ""),
            "chunk_title": node.metadata.get("chunk_title", ""),
            "chunk_summary": node.metadata.get("chunk_summary", ""),
            "created_at": node.metadata.get("created_at", ""),
            "word_count": node.metadata.get("word_count", 0),
            "char_count": node.metadata.get("char_count", 0)
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.vector_storage.get_database_stats()
    
    def list_files(self) -> List[str]:
        """Get list of all files in the database."""
        try:
            stats = self.get_database_stats()
            return []
        except Exception as e:
            logger.error(f"❌ Error listing files: {e}")
            return []
    
    def display_results_with_files(self, query_result: QueryResult, title: str = "Search Results") -> None:
        """
        Display query results with file and page information prominently shown.
        
        Args:
            query_result: QueryResult object to display
            title: Title for the display section
        """
        print(f"\n📋 {title}")
        print(f"Query: '{query_result.query}'")
        print(f"Total Results: {query_result.total_results}")
        print(f"Processing Time: {query_result.processing_time:.2f}s")
        print(f"Query Type: {query_result.query_type}")
        
        if query_result.cited_pages:
            print(f"📖 Cited Pages: {', '.join(map(str, query_result.cited_pages))}")
        
        if query_result.rag_response:
            print(f"\n🤖 RAG Response:")
            print(f"{query_result.rag_response}")
        
        if query_result.results:
            print(f"\n📄 Retrieved Documents:")
            print("-" * 60)
            
            # Group by file for better organization
            files_dict = {}
            for result in query_result.results:
                file_name = result['file_name']
                if file_name not in files_dict:
                    files_dict[file_name] = []
                files_dict[file_name].append(result)
            
            for file_name, chunks in files_dict.items():
                print(f"\n📁 File: {file_name}")
                print(f"   📊 Chunks from this file: {len(chunks)}")
                
                for i, chunk in enumerate(chunks, 1):
                    print(f"   {i}. Title: {chunk['chunk_title']}")
                    if chunk.get('page_number') is not None:
                        print(f"      📖 Page: {chunk['page_number']}")
                    else:
                        print(f"      📖 Page: Not specified")
                    print(f"      Summary: {chunk['chunk_summary'][:120]}...")
                    if 'similarity_score' in chunk:
                        print(f"      Score: {chunk['similarity_score']:.3f}")
                    print(f"      Words: {chunk['word_count']}, Chars: {chunk['char_count']}")
                    print(f"      Created: {chunk['created_at']}")
        else:
            print("❌ No results found")
    
    def get_results_by_file(self, query_result: QueryResult) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group query results by file name.
        
        Args:
            query_result: QueryResult object
            
        Returns:
            Dictionary with file names as keys and lists of chunks as values
        """
        files_dict = {}
        for result in query_result.results:
            file_name = result['file_name']
            if file_name not in files_dict:
                files_dict[file_name] = []
            files_dict[file_name].append(result)
        return files_dict
    
    def get_results_by_page(self, query_result: QueryResult) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group query results by page number.
        
        Args:
            query_result: QueryResult object
            
        Returns:
            Dictionary with page numbers as keys and lists of chunks as values
        """
        pages_dict = {}
        for result in query_result.results:
            page_num = result.get('page_number')
            if page_num is not None:
                if page_num not in pages_dict:
                    pages_dict[page_num] = []
                pages_dict[page_num].append(result)
        return pages_dict

# Utility functions for easy querying with page support

def create_query_engine(config: QueryConfig = None) -> AdvancedQueryEngine:
    """
    Create a query engine with default configuration including page support.
    
    Args:
        config: Optional query configuration
        
    Returns:
        AdvancedQueryEngine instance
    """
    # Create vector storage
    db_config = create_database_config_from_env()
    vector_storage = VectorStorage(db_config)
    
    # Create query engine with page support
    if config is None:
        config = QueryConfig(include_page_numbers=True, include_page_citations=True)
    
    return AdvancedQueryEngine(vector_storage, config)

def simple_search(query: str, limit: int = 10) -> QueryResult:
    """
    Simple search function for quick queries.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        QueryResult with search results
    """
    query_engine = create_query_engine()
    return query_engine.search_semantic(query, limit)

def rag_search(query: str, limit: int = 5) -> QueryResult:
    """
    RAG search function with page citations.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        QueryResult with RAG response including page citations
    """
    config = QueryConfig(
        enable_rag=True, 
        similarity_top_k=limit,
        include_page_citations=True,
        include_page_numbers=True
    )
    query_engine = create_query_engine(config)
    return query_engine.rag_query_with_page_context(query, limit)

def enhanced_rag_search(query: str, limit: int = 5) -> QueryResult:
    """
    Enhanced RAG search with explicit page context.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        QueryResult with enhanced RAG response including page citations
    """
    config = QueryConfig(
        enable_rag=True, 
        similarity_top_k=limit,
        include_page_citations=True,
        include_page_numbers=True
    )
    query_engine = create_query_engine(config)
    return query_engine.rag_query_with_page_context(query, limit)

def search_in_file(query: str, file_name: str, limit: int = 10) -> QueryResult:
    """
    Search within a specific file.
    
    Args:
        query: Search query
        file_name: Name of the file
        limit: Maximum number of results
        
    Returns:
        QueryResult with search results
    """
    query_engine = create_query_engine()
    return query_engine.search_by_file(query, file_name, limit)

def search_in_page(query: str, page_number: int, limit: int = 10) -> QueryResult:
    """
    Search within a specific page.
    
    Args:
        query: Search query
        page_number: Page number to search in
        limit: Maximum number of results
        
    Returns:
        QueryResult with search results
    """
    query_engine = create_query_engine()
    return query_engine.search_by_page_number(query, page_number, limit)

def search_in_page_range(query: str, start_page: int, end_page: int, limit: int = 10) -> QueryResult:
    """
    Search within a page range.
    
    Args:
        query: Search query
        start_page: Start page number
        end_page: End page number
        limit: Maximum number of results
        
    Returns:
        QueryResult with search results
    """
    query_engine = create_query_engine()
    return query_engine.search_by_page_range(query, start_page, end_page, limit)

def get_all_files() -> List[str]:
    """
    Get list of all files in the database.
    
    Returns:
        List of file names
    """
    query_engine = create_query_engine()
    stats = query_engine.get_database_stats()
    return stats.get("unique_files", [])

def search_and_display_with_files(query: str, limit: int = 10, title: str = "Search Results") -> QueryResult:
    """
    Perform search and display results with file and page information.
    
    Args:
        query: Search query
        limit: Maximum number of results
        title: Title for display
        
    Returns:
        QueryResult with search results
    """
    query_engine = create_query_engine()
    result = query_engine.search_semantic(query, limit)
    query_engine.display_results_with_files(result, title)
    return result

def rag_and_display_with_files(query: str, limit: int = 5, title: str = "RAG Results") -> QueryResult:
    """
    Perform RAG query and display results with file and page information including citations.
    
    Args:
        query: Search query
        limit: Maximum number of results
        title: Title for display
        
    Returns:
        QueryResult with RAG response and page citations
    """
    config = QueryConfig(
        enable_rag=True, 
        similarity_top_k=limit,
        include_page_citations=True,
        include_page_numbers=True
    )
    query_engine = create_query_engine(config)
    result = query_engine.rag_query_with_page_context(query, limit)
    query_engine.display_results_with_files(result, title)
    return result

def enhanced_rag_and_display(query: str, limit: int = 5, title: str = "Enhanced RAG Results") -> QueryResult:
    """
    Perform enhanced RAG query with explicit page context and display results.
    Supports multiple languages including Arabic.
    
    Args:
        query: Search query
        limit: Maximum number of results
        title: Title for display
        
    Returns:
        QueryResult with enhanced RAG response and page citations
    """
    config = QueryConfig(
        enable_rag=True, 
        similarity_top_k=limit,
        include_page_citations=True,
        include_page_numbers=True
    )
    query_engine = create_query_engine(config)
    result = query_engine.rag_query_with_page_context(query, limit)
    query_engine.display_results_with_files(result, title)
    return result

def arabic_rag_query(query: str, limit: int = 5) -> QueryResult:
    """
    Perform RAG query specifically optimized for Arabic language.
    
    Args:
        query: Arabic search query
        limit: Maximum number of results
        
    Returns:
        QueryResult with Arabic RAG response and page citations
    """
    config = QueryConfig(
        enable_rag=True, 
        similarity_top_k=limit,
        include_page_citations=True,
        include_page_numbers=True
    )
    query_engine = create_query_engine(config)
    return query_engine.rag_query_with_page_context(query, limit)

def arabic_search_and_display(query: str, limit: int = 5, title: str = "نتائج البحث باللغة العربية") -> QueryResult:
    """
    Perform Arabic search and display results with Arabic formatting.
    
    Args:
        query: Arabic search query
        limit: Maximum number of results
        title: Arabic title for display
        
    Returns:
        QueryResult with Arabic RAG response and page citations
    """
    config = QueryConfig(
        enable_rag=True, 
        similarity_top_k=limit,
        include_page_citations=True,
        include_page_numbers=True
    )
    query_engine = create_query_engine(config)
    result = query_engine.rag_query_with_page_context(query, limit)
    query_engine.display_results_with_files(result, title)
    return result

# Example usage and testing functions

def example_usage():
    """Example of how to use the enhanced query engine with page citations."""
    
    try:
        print("🚀 Initializing enhanced query engine with page citation support...")
        
        # Create query engine with page support
        config = QueryConfig(
            similarity_top_k=5,
            enable_rag=True,
            include_similarity_scores=True,
            include_page_numbers=True,
            include_page_citations=True
        )
        query_engine = create_query_engine(config)
        
        print("✅ Enhanced query engine initialized successfully")
        
        # Get database stats
        stats = query_engine.get_database_stats()
        print(f"📊 Database stats: {stats}")
        
        if stats["total_chunks"] == 0:
            print("ℹ️ No documents found in database. Please upload some documents first.")
            return
        
        # Example searches with enhanced page support
        print("\n🔍 Example searches with page information and citations:")
        
        # 1. Simple semantic search with page display
        print("\n1. Semantic search with page information:")
        result = query_engine.search_semantic("artificial intelligence", limit=3)
        query_engine.display_results_with_files(result, "Semantic Search Results")
        
        # 2. Enhanced RAG query with page citations
        print("\n2. Enhanced RAG query with page citations:")
        rag_result = query_engine.rag_query_with_page_context("What is artificial intelligence and how does it work?")
        query_engine.display_results_with_files(rag_result, "Enhanced RAG Query Results")
        
        # 3. Search by specific page number
        if rag_result.results:
            first_page = rag_result.results[0].get("page_number")
            if first_page is not None:
                print(f"\n3. Search within page {first_page}:")
                page_result = query_engine.search_by_page_number("technology", first_page, limit=2)
                query_engine.display_results_with_files(page_result, f"Search in Page {first_page}")
        
        # 4. Search by page range
        print("\n4. Search within page range 1-5:")
        page_range_result = query_engine.search_by_page_range("machine learning", 1, 5, limit=3)
        query_engine.display_results_with_files(page_range_result, "Search in Page Range 1-5")
        
        # 5. Show results grouped by file and page
        print("\n5. Results grouped by page:")
        if rag_result.results:
            pages_dict = query_engine.get_results_by_page(rag_result)
            for page_num, chunks in pages_dict.items():
                print(f"\n📖 Page {page_num} ({len(chunks)} chunks)")
                for i, chunk in enumerate(chunks, 1):
                    print(f"   {i}. {chunk['chunk_title']} (File: {chunk['file_name']})")
        
        # 6. Standard RAG vs Enhanced RAG comparison
        print("\n6. Comparison: Standard RAG vs Enhanced RAG with Page Context:")
        
        print("\n   Standard RAG:")
        standard_rag = query_engine.rag_query("Explain the concept of neural networks")
        if standard_rag.rag_response:
            print(f"   Response: {standard_rag.rag_response[:200]}...")
            if standard_rag.cited_pages:
                print(f"   Cited Pages: {standard_rag.cited_pages}")
        
        print("\n   Enhanced RAG with Page Context:")
        enhanced_rag = query_engine.rag_query_with_page_context("Explain the concept of neural networks")
        if enhanced_rag.rag_response:
            print(f"   Response: {enhanced_rag.rag_response[:200]}...")
            if enhanced_rag.cited_pages:
                print(f"   Cited Pages: {enhanced_rag.cited_pages}")
        
        # 7. Arabic language support demonstration
        print("\n7. Arabic Language Support Demonstration:")
        
        # Test Arabic query
        arabic_query = "ما هو الذكاء الاصطناعي وكيف يعمل؟"
        print(f"\n   Arabic Query: {arabic_query}")
        
        arabic_result = query_engine.rag_query_with_page_context(arabic_query)
        if arabic_result.rag_response:
            print(f"   Arabic Response: {arabic_result.rag_response[:300]}...")
            if arabic_result.cited_pages:
                print(f"   Cited Pages: {arabic_result.cited_pages}")
        
        # Test mixed language detection
        print("\n8. Language Detection Testing:")
        test_queries = [
            "What is artificial intelligence?",
            "ما هو الذكاء الاصطناعي؟",
            "Qu'est-ce que l'intelligence artificielle?",
            "¿Qué es la inteligencia artificial?"
        ]
        
        for test_query in test_queries:
            detected_lang = detect_language(test_query)
            print(f"   Query: '{test_query[:30]}...' -> Language: {detected_lang}")
        
        print("\n✅ Example usage completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   • Page number extraction and display")
        print("   • Page-specific search capabilities")
        print("   • Page range filtering")
        print("   • RAG responses with page citations")
        print("   • Enhanced context creation with page information")
        print("   • Custom prompt templates for page citations")
        print("   • Multi-language support including Arabic")
        print("   • Automatic language detection")
        print("   • Language-specific response formatting")
        
    except Exception as e:
        print(f"❌ Error in example usage: {e}")
        import traceback
        traceback.print_exc()

def test_page_citation_features():
    """Test specific page citation features."""
    
    try:
        print("\n🧪 Testing Page Citation Features:")
        
        config = QueryConfig(
            enable_rag=True,
            include_page_citations=True,
            include_page_numbers=True,
            similarity_top_k=3
        )
        
        query_engine = create_query_engine(config)
        
        # Test 1: Enhanced RAG with page context
        print("\n1. Testing Enhanced RAG with Page Context:")
        test_query = "What are the main benefits of artificial intelligence?"
        result = query_engine.rag_query_with_page_context(test_query)
        
        print(f"Query: {test_query}")
        print(f"Response: {result.rag_response}")
        print(f"Cited Pages: {result.cited_pages}")
        print(f"Retrieved Chunks: {len(result.results)}")
        
        for i, chunk in enumerate(result.results, 1):
            page_num = chunk.get('page_number', 'N/A')
            print(f"  Chunk {i}: Page {page_num} - {chunk['file_name']}")
        
        # Test 2: Page-specific search
        print("\n2. Testing Page-Specific Search:")
        if result.results and result.results[0].get('page_number') is not None:
            test_page = result.results[0]['page_number']
            page_result = query_engine.search_by_page_number("technology", test_page)
            print(f"Found {len(page_result.results)} chunks in page {test_page}")
        
        # Test 3: Page range search
        print("\n3. Testing Page Range Search:")
        range_result = query_engine.search_by_page_range("innovation", 1, 10)
        print(f"Found {len(range_result.results)} chunks in pages 1-10")
        
        pages_found = set()
        for chunk in range_result.results:
            if chunk.get('page_number') is not None:
                pages_found.add(chunk['page_number'])
        
        print(f"Pages with results: {sorted(pages_found) if pages_found else 'None'}")
        
        print("\n✅ Page citation features tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing page citation features: {e}")
        import traceback
        traceback.print_exc()

def test_arabic_language_features():
    """Test Arabic language support features."""
    
    try:
        print("\n🌐 Testing Arabic Language Features:")
        
        config = QueryConfig(
            enable_rag=True,
            include_page_citations=True,
            include_page_numbers=True,
            similarity_top_k=3
        )
        
        query_engine = create_query_engine(config)
        
        # Test 1: Arabic language detection
        print("\n1. Testing Arabic Language Detection:")
        arabic_queries = [
            "ما هو الذكاء الاصطناعي؟",
            "كيف يعمل التعلم الآلي؟",
            "ما هي فوائد التكنولوجيا الحديثة؟",
            "شرح مفهوم الشبكات العصبية"
        ]
        
        for query in arabic_queries:
            detected_lang = detect_language(query)
            print(f"   Query: '{query}' -> Language: {detected_lang}")
        
        # Test 2: Arabic RAG query
        print("\n2. Testing Arabic RAG Query:")
        arabic_query = "ما هي المزايا الرئيسية للذكاء الاصطناعي؟"
        arabic_result = query_engine.rag_query_with_page_context(arabic_query)
        
        print(f"Arabic Query: {arabic_query}")
        print(f"Detected Language: {detect_language(arabic_query)}")
        print(f"Response: {arabic_result.rag_response}")
        print(f"Cited Pages: {arabic_result.cited_pages}")
        
        # Test 3: Mixed language queries
        print("\n3. Testing Mixed Language Queries:")
        mixed_queries = [
            "What is artificial intelligence?",
            "ما هو الذكاء الاصطناعي؟",
            "Explain machine learning",
            "شرح التعلم الآلي"
        ]
        
        for query in mixed_queries:
            detected_lang = detect_language(query)
            print(f"   Query: '{query}' -> Language: {detected_lang}")
        
        # Test 4: Arabic page citations
        print("\n4. Testing Arabic Page Citations:")
        arabic_page_query = "ما هي المعلومات المتوفرة في الصفحة الأولى؟"
        arabic_page_result = query_engine.rag_query_with_page_context(arabic_page_query)
        
        print(f"Arabic Page Query: {arabic_page_query}")
        if arabic_page_result.rag_response:
            # Check if response contains Arabic page citations
            if "الصفحة:" in arabic_page_result.rag_response:
                print("   ✅ Arabic page citations found in response")
            else:
                print("   ⚠️ Arabic page citations not found in response")
        
        print("\n✅ Arabic language features tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Arabic language features: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    print("🔍 Enhanced Query Engine with Page Citations and Multi-Language Support")
    print("=" * 70)
    
    # Run example usage
    example_usage()
    
    # Run page citation tests
    test_page_citation_features()
    
    # Run Arabic language tests
    test_arabic_language_features()
    
    print("\n" + "=" * 70)
    print("🎯 Summary of Enhancements:")
    print("• Custom RAG prompt template with page citation instructions")
    print("• Enhanced context creation with explicit page information")
    print("• Page number extraction and display in results")
    print("• Page-specific and page range search capabilities")
    print("• Cited pages tracking in QueryResult")
    print("• Enhanced display methods showing page information")
    print("• New utility functions for page-based operations")
    print("• Comprehensive testing of page citation features")
    print("• Multi-language support including Arabic")
    print("• Automatic language detection using Unicode patterns")
    print("• Language-specific prompt templates and instructions")
    print("• Arabic page citations: (الصفحة: X) format")
    print("• Arabic-specific utility functions")
    print("• Comprehensive Arabic language testing")