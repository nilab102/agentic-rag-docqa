import re
from typing import List, Dict, Any, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import os

class BoundaryType(Enum):
    """Types of semantic boundaries in the text"""
    CHAPTER = "CHAPTER"
    SECTION = "SECTION"
    SUBSECTION = "SUBSECTION"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    LIST = "LIST"
    PAGE = "PAGE"
    FIGURE = "FIGURE"
    CODE = "CODE"
    QUOTE = "QUOTE"

@dataclass
class Page:
    """Represents a single page with its content"""
    content: str
    page_number: int
    start_pos: int
    end_pos: int
    word_count: int = 0
    
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split())

@dataclass
class TextSegment:
    """Represents a segment of text with its semantic information"""
    content: str
    start_pos: int
    end_pos: int
    boundary_type: BoundaryType
    title: Optional[str] = None
    page_numbers: List[int] = None
    word_count: int = 0
    is_complete: bool = True  # Whether the segment is complete (not split)
    
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.page_numbers is None:
            self.page_numbers = []

@dataclass
class Chunk:
    """Represents a final chunk with metadata"""
    content: str
    chunk_id: int
    pages: List[Page]
    segments: List[TextSegment]
    word_count: int
    page_count: int
    page_numbers: List[int]
    boundary_info: Dict[str, Any]
    overlap_pages: List[int] = None  # Pages that are overlapped from previous chunk
    
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.overlap_pages is None:
            self.overlap_pages = []

class SmartTextChunker:
    """
    Intelligent text chunker that respects page boundaries and maintains overlap.
    Always chunks after page breaks and maintains one page overlap.
    """
    
    def __init__(self, 
                 target_pages_per_chunk: int = 5,
                 overlap_pages: int = 1,
                 max_pages_per_chunk: int = 8,
                 min_pages_per_chunk: int = 2,
                 respect_boundaries: bool = True):
        """
        Initialize the smart text chunker.
        
        Args:
            target_pages_per_chunk: Target number of pages per chunk
            overlap_pages: Number of pages to overlap between chunks
            max_pages_per_chunk: Maximum pages allowed in a chunk
            min_pages_per_chunk: Minimum pages required in a chunk
            respect_boundaries: Whether to respect semantic boundaries within pages
        """
        self.target_pages_per_chunk = target_pages_per_chunk
        self.overlap_pages = overlap_pages
        self.max_pages_per_chunk = max_pages_per_chunk
        self.min_pages_per_chunk = min_pages_per_chunk
        self.respect_boundaries = respect_boundaries
        
        # Updated patterns for finding page boundaries with manual page numbering
        self.page_patterns = [
            r'PAGE_NUMBER:\s*(\d+)\s*\n\[PAGE_START\](.*?)(?=PAGE_NUMBER:\s*\d+\s*\n\[PAGE_START\]|\[PAGE_END\]|$)',  # Manual page numbering format
            r'\[PAGE_START\](.*?)(?=\[PAGE_START\]|\[PAGE_END\]|$)',  # Simple page markers (fallback)
            r'\[PAGE_START\](.*?)\[PAGE_BREAK\]',  # Legacy format for backward compatibility
            r'(?:^|\n).*?Page\s+(\d+).*?\n(.*?)(?=Page\s+\d+|$)',  # Traditional page numbering
        ]
        
        # Boundary patterns for semantic markers within pages
        self.boundary_patterns = {
            BoundaryType.CHAPTER: {
                'start': r'\[CHAPTER_START:\s*([^\]]+)\]',
                'end': r'\[CHAPTER_END[^\]]*\]',
                'priority': 1
            },
            BoundaryType.SECTION: {
                'start': r'\[SECTION_START:\s*([^\]]+)\]',
                'end': r'\[SECTION_END[^\]]*\]',
                'priority': 2
            },
            BoundaryType.SUBSECTION: {
                'start': r'\[SUBSECTION_START:\s*([^\]]+)\]',
                'end': r'\[SUBSECTION_END[^\]]*\]',
                'priority': 3
            },
            BoundaryType.TABLE: {
                'start': r'\[TABLE_START[^\]]*\]',
                'end': r'\[TABLE_END\]',
                'priority': 1
            },
            BoundaryType.PARAGRAPH: {
                'start': r'\[PARAGRAPH_START\]',
                'end': r'\[PARAGRAPH_END\]',
                'priority': 4
            },
            BoundaryType.LIST: {
                'start': r'\[LIST_START\]',
                'end': r'\[LIST_END\]',
                'priority': 3
            },
            BoundaryType.FIGURE: {
                'start': r'\[FIGURE_START:\s*([^\]]+)\]',
                'end': r'\[FIGURE_END\]',
                'priority': 2
            },
            BoundaryType.CODE: {
                'start': r'\[CODE_START[^\]]*\]',
                'end': r'\[CODE_END\]',
                'priority': 2
            },
            BoundaryType.QUOTE: {
                'start': r'\[QUOTE_START\]',
                'end': r'\[QUOTE_END\]',
                'priority': 3
            }
        }
    
    def extract_pages(self, text: str) -> List[Page]:
        """Extract pages from text based on page markers."""
        pages = []
        
        # Method 1: Try to find pages with manual page numbering format
        page_pattern = r'PAGE_NUMBER:\s*(\d+)\s*\n\[PAGE_START\](.*?)(?=PAGE_NUMBER:\s*\d+\s*\n\[PAGE_START\]|\[PAGE_END\]|$)'
        page_matches = list(re.finditer(page_pattern, text, re.DOTALL))
        
        if page_matches:
            for match in page_matches:
                page_num_str = match.group(1)
                content = match.group(2).strip()
                
                if content:  # Only add non-empty pages
                    try:
                        page_num = int(page_num_str)
                        
                        page = Page(
                            content=content,
                            page_number=page_num,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            word_count=len(content.split())
                        )
                        pages.append(page)
                    except ValueError:
                        print(f"Warning: Could not parse page number '{page_num_str}'")
                        continue
        else:
            # Method 2: Try simple [PAGE_START] format without PAGE_NUMBER header
            simple_pattern = r'\[PAGE_START\](.*?)(?=\[PAGE_START\]|\[PAGE_END\]|$)'
            simple_matches = list(re.finditer(simple_pattern, text, re.DOTALL))
            
            if simple_matches:
                for i, match in enumerate(simple_matches):
                    content = match.group(1).strip()
                    if content and len(content.split()) > 10:  # Only meaningful pages
                        page_num = self.extract_page_number_from_content(content)
                        if page_num is None:
                            page_num = i + 1  # Fallback to sequential numbering
                        
                        page = Page(
                            content=content,
                            page_number=page_num,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            word_count=len(content.split())
                        )
                        pages.append(page)
            else:
                # Method 3: Split by page breaks or double newlines as fallback
                page_breaks = re.split(r'\[PAGE_BREAK\]|\n\s*\n\s*\n', text)
                for i, page_content in enumerate(page_breaks):
                    content = page_content.strip()
                    if content and len(content.split()) > 10:  # Only meaningful pages
                        page_num = self.extract_page_number_from_content(content)
                        if page_num is None:
                            page_num = i + 1
                        
                        page = Page(
                            content=content,
                            page_number=page_num,
                            start_pos=0,  # Approximate since we split the text
                            end_pos=len(content),
                            word_count=len(content.split())
                        )
                        pages.append(page)
        
        # Sort pages by page number
        pages.sort(key=lambda x: x.page_number)
        return pages
    
    def extract_page_number_from_content(self, content: str) -> Optional[int]:
        """Extract page number from page content."""
        # Look for various page number patterns
        patterns = [
            r'PAGE_NUMBER:\s*(\d+)',  # New format with PAGE_NUMBER header
            r'Page\s+(\d+)',
            r'^\s*(\d+)\s*$',
            r'- (\d+) -',
            r'\[(\d+)\]',
            r'(?:^|\n)\s*(\d+)\s*(?:\n|$)',
            # Legacy patterns for backward compatibility
            r'\[PAGE_START:\s*-(\d+)\]',
            r'\[PAGE_END:\s*-(\d+)\]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            if matches:
                try:
                    page_num = int(matches[0])
                    # If it's a negative number (from legacy format), convert to positive
                    return abs(page_num)
                except ValueError:
                    continue
        
        return None
    
    def parse_segments_in_page(self, page: Page) -> List[TextSegment]:
        """Parse semantic segments within a single page."""
        segments = []
        content = page.content
        
        # Find all semantic boundaries within this page
        all_boundaries = []
        
        for boundary_type, patterns in self.boundary_patterns.items():
            start_pattern = patterns['start']
            end_pattern = patterns['end']
            priority = patterns['priority']
            
            # Find start markers
            for match in re.finditer(start_pattern, content):
                title = match.group(1) if match.groups() else None
                all_boundaries.append({
                    'type': 'start',
                    'boundary_type': boundary_type,
                    'position': match.start(),
                    'end_position': match.end(),
                    'priority': priority,
                    'title': title
                })
            
            # Find end markers
            for match in re.finditer(end_pattern, content):
                all_boundaries.append({
                    'type': 'end',
                    'boundary_type': boundary_type,
                    'position': match.start(),
                    'end_position': match.end(),
                    'priority': priority
                })
        
        # Sort boundaries by position
        all_boundaries.sort(key=lambda x: x['position'])
        
        # Build segments by matching start/end pairs
        boundary_stack = []
        
        for boundary in all_boundaries:
            if boundary['type'] == 'start':
                boundary_stack.append(boundary)
            elif boundary['type'] == 'end':
                # Find matching start
                matching_start = None
                for i in range(len(boundary_stack) - 1, -1, -1):
                    if boundary_stack[i]['boundary_type'] == boundary['boundary_type']:
                        matching_start = boundary_stack.pop(i)
                        break
                
                if matching_start:
                    # Create segment
                    start_pos = matching_start['end_position']
                    end_pos = boundary['position']
                    segment_content = content[start_pos:end_pos].strip()
                    
                    if segment_content:
                        segment = TextSegment(
                            content=segment_content,
                            start_pos=page.start_pos + start_pos,
                            end_pos=page.start_pos + end_pos,
                            boundary_type=boundary['boundary_type'],
                            title=matching_start.get('title'),
                            page_numbers=[page.page_number],
                            word_count=len(segment_content.split())
                        )
                        segments.append(segment)
        
        # If no segments found, treat entire page as a paragraph
        if not segments:
            segment = TextSegment(
                content=content,
                start_pos=page.start_pos,
                end_pos=page.end_pos,
                boundary_type=BoundaryType.PARAGRAPH,
                page_numbers=[page.page_number],
                word_count=page.word_count
            )
            segments.append(segment)
        
        return segments
    
    def create_chunks_from_pages(self, pages: List[Page]) -> List[Chunk]:
        """Create chunks from pages with overlap."""
        if not pages:
            return []
        
        chunks = []
        chunk_id = 1
        i = 0
        
        while i < len(pages):
            # Determine chunk size
            chunk_pages = []
            pages_to_include = min(self.target_pages_per_chunk, len(pages) - i)
            
            # Add main pages for this chunk
            for j in range(pages_to_include):
                if i + j < len(pages):
                    chunk_pages.append(pages[i + j])
            
            # Add overlap pages from next chunk if not the last chunk
            overlap_pages_list = []
            if i + pages_to_include < len(pages):  # Not the last chunk
                overlap_end = min(i + pages_to_include + self.overlap_pages, len(pages))
                for j in range(i + pages_to_include, overlap_end):
                    chunk_pages.append(pages[j])
                    overlap_pages_list.append(pages[j].page_number)
            
            # Create chunk
            chunk = self.create_chunk_from_pages(chunk_pages, chunk_id, overlap_pages_list)
            chunks.append(chunk)
            
            chunk_id += 1
            i += pages_to_include  # Move by non-overlapping pages only
        
        return chunks
    
    def create_chunk_from_pages(self, pages: List[Page], chunk_id: int, overlap_page_numbers: List[int]) -> Chunk:
        """Create a chunk from a list of pages."""
        # Combine page contents
        content_parts = []
        all_segments = []
        page_numbers = []
        
        for page in pages:
            content_parts.append(f"[PAGE {page.page_number} START]\n{page.content}\n[PAGE {page.page_number} END]")
            page_numbers.append(page.page_number)
            
            # Parse segments within this page if needed
            if self.respect_boundaries:
                segments = self.parse_segments_in_page(page)
                all_segments.extend(segments)
            else:
                # Create a single segment for the entire page
                segment = TextSegment(
                    content=page.content,
                    start_pos=page.start_pos,
                    end_pos=page.end_pos,
                    boundary_type=BoundaryType.PAGE,
                    page_numbers=[page.page_number],
                    word_count=page.word_count
                )
                all_segments.append(segment)
        
        # Join content with clear page separators
        full_content = '\n\n'.join(content_parts)
        
        # Calculate metrics
        total_words = sum(page.word_count for page in pages)
        
        # Create boundary info
        boundary_info = {
            'starts_with_page': pages[0].page_number if pages else None,
            'ends_with_page': pages[-1].page_number if pages else None,
            'contains_tables': any(seg.boundary_type == BoundaryType.TABLE for seg in all_segments),
            'contains_figures': any(seg.boundary_type == BoundaryType.FIGURE for seg in all_segments),
            'contains_chapters': any(seg.boundary_type == BoundaryType.CHAPTER for seg in all_segments),
            'segment_titles': [seg.title for seg in all_segments if seg.title],
            'page_boundaries': True,  # Always true for this chunker
            'overlap_info': f"Overlaps {len(overlap_page_numbers)} pages: {overlap_page_numbers}" if overlap_page_numbers else "No overlap"
        }
        
        return Chunk(
            content=full_content,
            chunk_id=chunk_id,
            pages=pages,
            segments=all_segments,
            word_count=total_words,
            page_count=len(pages),
            page_numbers=page_numbers,
            boundary_info=boundary_info,
            overlap_pages=overlap_page_numbers
        )
    
    def chunk_text(self, extracted_text: str, 
                   target_pages: Optional[int] = None,
                   overlap_pages: Optional[int] = None) -> List[Chunk]:
        """
        Main method to chunk extracted text by page boundaries with overlap.
        
        Args:
            extracted_text: Text with page markers from document processor
            target_pages: Target pages per chunk (overrides default)
            overlap_pages: Pages to overlap between chunks (overrides default)
            
        Returns:
            List of chunks with page-based boundaries and overlap
        """
        # Update parameters if provided
        if target_pages:
            self.target_pages_per_chunk = target_pages
        
        if overlap_pages is not None:
            self.overlap_pages = overlap_pages
        
        print(f"Chunking text with page-based boundaries:")
        print(f"  Target pages per chunk: {self.target_pages_per_chunk}")
        print(f"  Overlap pages: {self.overlap_pages}")
        print(f"  Respect semantic boundaries: {self.respect_boundaries}")
        
        # Extract pages from text
        print("\nExtracting pages from text...")
        pages = self.extract_pages(extracted_text)
        print(f"Found {len(pages)} pages")
        
        if not pages:
            print("Warning: No pages found in text. Check page markers.")
            return []
        
        # Print page summary
        print(f"Page numbers found: {sorted([p.page_number for p in pages])}")
        print(f"Average words per page: {sum(p.word_count for p in pages) // len(pages)}")
        
        # Create chunks with page boundaries and overlap
        print("\nCreating page-based chunks with overlap...")
        chunks = self.create_chunks_from_pages(pages)
        
        # Print summary
        self.print_chunking_summary(chunks)
        
        return chunks
    
    def print_chunking_summary(self, chunks: List[Chunk]) -> None:
        """Print a summary of the chunking results."""
        print(f"\n{'='*70}")
        print("PAGE-BASED CHUNKING SUMMARY")
        print(f"{'='*70}")
        print(f"Total chunks created: {len(chunks)}")
        
        total_words = sum(chunk.word_count for chunk in chunks)
        total_pages = sum(len(chunk.pages) for chunk in chunks)
        unique_pages = len(set(p for chunk in chunks for p in chunk.page_numbers))
        
        print(f"Total words: {total_words:,}")
        print(f"Total page instances: {total_pages} (including overlaps)")
        print(f"Unique pages: {unique_pages}")
        print(f"Average words per chunk: {total_words // len(chunks) if chunks else 0:,}")
        print(f"Average pages per chunk: {total_pages / len(chunks):.1f}")
        
        print(f"\nDetailed Chunk Information:")
        for chunk in chunks:
            main_pages = [p for p in chunk.page_numbers if p not in chunk.overlap_pages]
            overlap_info = f" + {len(chunk.overlap_pages)} overlap" if chunk.overlap_pages else ""
            
            print(f"  Chunk {chunk.chunk_id}: {chunk.word_count:,} words")
            print(f"    Pages: {main_pages}{overlap_info}")
            print(f"    Boundaries: {chunk.boundary_info['starts_with_page']} → {chunk.boundary_info['ends_with_page']}")
            
            if chunk.boundary_info['segment_titles']:
                titles = [t for t in chunk.boundary_info['segment_titles'] if t][:2]
                if titles:
                    print(f"    Titles: {', '.join(titles)}{'...' if len(chunk.boundary_info['segment_titles']) > 2 else ''}")
            
            special_content = []
            if chunk.boundary_info['contains_tables']:
                special_content.append("Tables")
            if chunk.boundary_info['contains_figures']:
                special_content.append("Figures")
            if chunk.boundary_info['contains_chapters']:
                special_content.append("Chapters")
            
            if special_content:
                print(f"    Contains: {', '.join(special_content)}")
            print()
    
    def save_chunks(self, chunks: List[Chunk], output_dir: str = "chunks") -> Dict[str, str]:
        """Save chunks to separate files without metadata header."""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        for chunk in chunks:
            filename = f"chunk_{chunk.chunk_id:03d}_pages_{chunk.page_numbers[0]}-{chunk.page_numbers[-1]}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Save only the chunk content without metadata header
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(chunk.content)
            
            saved_files[f"chunk_{chunk.chunk_id}"] = filepath
        
        return saved_files

# Example usage function
def example_usage():
    """Example of how to use the improved SmartTextChunker."""
    
    # Specify the input file path - updated to use the Saudi Arabia findings file
    input_file = "/Users/nilab/Desktop/projects/agentic-rag-docqa/utilites/Saudi-Arabia-findings_extracted.txt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please ensure the file exists and try again.")
        return
    
    # Read the text from file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            extracted_text = f.read()
        print(f"Successfully loaded text file: {len(extracted_text):,} characters")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return
    
    # Initialize improved chunker with page-based boundaries
    chunker = SmartTextChunker(
        target_pages_per_chunk=3,  # Target 3 pages per chunk
        overlap_pages=1,           # 1 page overlap between chunks
        max_pages_per_chunk=5,     # Maximum 5 pages per chunk
        min_pages_per_chunk=1,     # Minimum 1 page per chunk
        respect_boundaries=True    # Respect semantic boundaries within pages
    )
    
    print("Initialized improved SmartTextChunker for page-based chunking with overlap")
    
    # Chunk the text
    chunks = chunker.chunk_text(extracted_text)
    
    if not chunks:
        print("No chunks were created. Please check your input text format.")
        return
    
    # Save chunks
    saved_files = chunker.save_chunks(chunks, "saudi_arabia_chunks")
    
    print(f"\nSuccessfully saved {len(saved_files)} chunk files:")
    for chunk_name, filepath in saved_files.items():
        print(f"  {chunk_name}: {filepath}")
    
    # Additional analysis
    print(f"\nChunking Analysis:")
    print(f"  Input file: {input_file}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Page-based boundaries: Always after page breaks")
    print(f"  Overlap strategy: {chunker.overlap_pages} page(s) between chunks")
    print(f"  Semantic awareness: {'Enabled' if chunker.respect_boundaries else 'Disabled'}")
    print(f"  Page numbers: {[chunk.page_numbers for chunk in chunks]}")
    return chunks

def chunk_saudi_arabia_findings():
    """Simple function to chunk the Saudi Arabia findings document."""
    
    # File path
    input_file = "Saudi-Arabia-findings_extracted.txt"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return None
    
    # Read the text
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            extracted_text = f.read()
        print(f"✓ Loaded Saudi Arabia findings: {len(extracted_text):,} characters")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    # Initialize chunker with updated settings
    chunker = SmartTextChunker(
        target_pages_per_chunk=5,  # Updated to match user's preference
        overlap_pages=1,
        max_pages_per_chunk=5,
        min_pages_per_chunk=1,
        respect_boundaries=True
    )
    
    # Chunk the text
    chunks = chunker.chunk_text(extracted_text)
    
    if not chunks:
        print("No chunks created.")
        return None
    
    # Save chunks (now without metadata)
    saved_files = chunker.save_chunks(chunks, "saudi_arabia_chunks")
    
    print(f"✓ Created {len(chunks)} chunks without metadata")
    print(f"✓ Saved to directory: saudi_arabia_chunks/")
    
    return chunks

if __name__ == "__main__":
    example_usage()