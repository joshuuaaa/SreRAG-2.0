"""Enhanced document processor for medical emergency texts with smart chunking"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class DocumentChunk:
    """Structured document chunk with metadata"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    section: str
    emergency_type: str
    severity_level: str
    keywords: List[str]

class MedicalDocumentProcessor:
    """Advanced processor for medical emergency documents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('chunk_tokens', 300)
        self.overlap_size = config.get('chunk_overlap_tokens', 40)
        
        # Medical emergency patterns
        self.emergency_patterns = {
            'bleeding': [
                r'bleeding', r'hemorrhage', r'blood loss', r'arterial', r'venous',
                r'tourniquet', r'pressure', r'wound', r'laceration'
            ],
            'cardiac': [
                r'heart attack', r'cardiac arrest', r'chest pain', r'myocardial',
                r'arrhythmia', r'cpr', r'defibrillation', r'pulse'
            ],
            'respiratory': [
                r'breathing', r'airway', r'choking', r'asthma', r'pneumonia',
                r'oxygen', r'ventilation', r'respiratory'
            ],
            'neurological': [
                r'stroke', r'seizure', r'unconscious', r'head injury', r'concussion',
                r'spinal', r'neurological', r'consciousness'
            ],
            'trauma': [
                r'fracture', r'burn', r'injury', r'accident', r'trauma',
                r'wound', r'laceration', r'contusion'
            ],
            'poisoning': [
                r'poison', r'overdose', r'toxic', r'antidote', r'ingestion',
                r'chemical', r'drug'
            ]
        }
        
        self.severity_patterns = {
            'critical': [
                r'life-threatening', r'critical', r'severe', r'emergency',
                r'immediate', r'urgent', r'massive', r'major'
            ],
            'moderate': [
                r'moderate', r'significant', r'substantial', r'concerning'
            ],
            'mild': [
                r'mild', r'minor', r'slight', r'small', r'limited'
            ]
        }
        
        self.section_patterns = {
            'overview': [r'overview', r'introduction', r'definition'],
            'assessment': [r'assessment', r'evaluation', r'diagnosis', r'signs', r'symptoms'],
            'treatment': [r'treatment', r'management', r'intervention', r'therapy'],
            'procedure': [r'procedure', r'protocol', r'steps', r'technique'],
            'complications': [r'complications', r'adverse', r'contraindications'],
            'special_populations': [r'pediatric', r'geriatric', r'pregnancy', r'infant']
        }
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a single medical document into structured chunks"""
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract document metadata
        doc_metadata = self._extract_document_metadata(file_path, content)
        
        # Split into sections based on headers
        sections = self._split_into_sections(content)
        
        chunks = []
        for section_title, section_content in sections:
            # Create chunks from this section
            section_chunks = self._create_smart_chunks(
                section_content, 
                section_title,
                doc_metadata,
                file_path
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _extract_document_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract metadata from document"""
        filename = Path(file_path).name
        
        # Detect emergency type from filename and content
        emergency_type = self._detect_emergency_type(content)
        
        # Extract source information
        source_match = re.search(r'Source:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        source = source_match.group(1) if source_match else filename
        
        # Extract date/version
        date_match = re.search(r'(?:Updated|Date|Version):\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        date = date_match.group(1) if date_match else 'Unknown'
        
        return {
            'filename': filename,
            'source': source,
            'date': date,
            'emergency_type': emergency_type,
            'file_path': file_path
        }
    
    def _detect_emergency_type(self, content: str) -> str:
        """Detect primary emergency type from content"""
        content_lower = content.lower()
        
        type_scores = {}
        for emergency_type, patterns in self.emergency_patterns.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, content_lower))
            type_scores[emergency_type] = score
        
        if not type_scores or max(type_scores.values()) == 0:
            return 'general'
        
        return max(type_scores, key=type_scores.get)
    
    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split document into logical sections"""
        
        # Headers patterns (===, SCENARIO, numbered sections, etc.)
        header_patterns = [
            r'^={3,}$\n(.+?)\n^={3,}$',  # === TITLE ===
            r'^(SCENARIO \d+[A-Z]*:.*?)$',  # SCENARIO 1A: Title
            r'^(\d+\.\s+[A-Z][^.]*?)$',   # 1. Section Title
            r'^([A-Z][A-Z\s]{3,}?)$',     # ALL CAPS HEADERS
        ]
        
        sections = []
        current_section = ""
        current_title = "Introduction"
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this line is a header
            is_header = False
            for pattern in header_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # Save previous section
                    if current_section.strip():
                        sections.append((current_title, current_section.strip()))
                    
                    # Start new section
                    current_title = line
                    current_section = ""
                    is_header = True
                    break
            
            if not is_header:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            sections.append((current_title, current_section.strip()))
        
        # If no sections found, treat entire content as one section
        if not sections:
            sections = [("Full Document", content)]
        
        return sections
    
    def _create_smart_chunks(self, content: str, section_title: str, 
                           doc_metadata: Dict[str, Any], file_path: str) -> List[DocumentChunk]:
        """Create semantically coherent chunks from section content"""
        
        # Split on natural boundaries (paragraphs, lists, etc.)
        paragraphs = self._split_on_boundaries(content)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph.split())
            
            # If paragraph alone exceeds chunk size, split it
            if para_size > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk = self._create_chunk(
                        current_chunk, section_title, doc_metadata, file_path, len(chunks)
                    )
                    chunks.append(chunk)
                    current_chunk = ""
                    current_size = 0
                
                # Split large paragraph
                sub_chunks = self._split_large_paragraph(paragraph)
                for sub_chunk in sub_chunks:
                    chunk = self._create_chunk(
                        sub_chunk, section_title, doc_metadata, file_path, len(chunks)
                    )
                    chunks.append(chunk)
            
            # If adding this paragraph would exceed chunk size
            elif current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk with overlap
                chunk = self._create_chunk(
                    current_chunk, section_title, doc_metadata, file_path, len(chunks)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n" + paragraph
                current_size = len(current_chunk.split())
            
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size = len(current_chunk.split())
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk, section_title, doc_metadata, file_path, len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_on_boundaries(self, content: str) -> List[str]:
        """Split content on natural boundaries"""
        
        # Split on double newlines first (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        
        refined_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Further split on list items or numbered steps
            if re.search(r'^\d+\.', para, re.MULTILINE):
                # Split on numbered items
                items = re.split(r'(?=^\d+\.)', para, flags=re.MULTILINE)
                refined_paragraphs.extend([item.strip() for item in items if item.strip()])
            elif re.search(r'^[-•*]', para, re.MULTILINE):
                # Split on bullet points
                items = re.split(r'(?=^[-•*])', para, flags=re.MULTILINE)
                refined_paragraphs.extend([item.strip() for item in items if item.strip()])
            else:
                refined_paragraphs.append(para)
        
        return refined_paragraphs
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph into smaller chunks"""
        sentences = re.split(r'[.!?]+', paragraph)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_size = sentence_size
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_size = len(current_chunk.split())
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text for chunk continuity"""
        words = text.split()
        if len(words) <= self.overlap_size:
            return text
        
        return ' '.join(words[-self.overlap_size:])
    
    def _create_chunk(self, text: str, section_title: str, 
                     doc_metadata: Dict[str, Any], file_path: str, chunk_index: int) -> DocumentChunk:
        """Create a DocumentChunk object with full metadata"""
        
        # Generate unique chunk ID
        chunk_id = hashlib.md5(f"{file_path}_{chunk_index}_{text[:50]}".encode()).hexdigest()[:12]
        
        # Detect section type
        section_type = self._classify_section(section_title, text)
        
        # Detect severity level
        severity = self._detect_severity(text)
        
        # Extract keywords
        keywords = self._extract_keywords(text)
        
        metadata = {
            **doc_metadata,
            'section_title': section_title,
            'section_type': section_type,
            'chunk_index': chunk_index,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        return DocumentChunk(
            text=text.strip(),
            metadata=metadata,
            chunk_id=chunk_id,
            source=doc_metadata['source'],
            section=section_type,
            emergency_type=doc_metadata['emergency_type'],
            severity_level=severity,
            keywords=keywords
        )
    
    def _classify_section(self, section_title: str, content: str) -> str:
        """Classify section type based on title and content"""
        title_lower = section_title.lower()
        content_lower = content.lower()
        
        section_scores = {}
        for section_type, patterns in self.section_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, title_lower):
                    score += 3  # Title matches are weighted higher
                score += len(re.findall(pattern, content_lower))
            section_scores[section_type] = score
        
        if not section_scores or max(section_scores.values()) == 0:
            return 'general'
        
        return max(section_scores, key=section_scores.get)
    
    def _detect_severity(self, text: str) -> str:
        """Detect severity level from text content"""
        text_lower = text.lower()
        
        severity_scores = {}
        for severity, patterns in self.severity_patterns.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, text_lower))
            severity_scores[severity] = score
        
        if not severity_scores or max(severity_scores.values()) == 0:
            return 'unknown'
        
        return max(severity_scores, key=severity_scores.get)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract medical keywords from text"""
        # Common medical terms to extract
        medical_terms = []
        
        # Extract all emergency-related terms
        for patterns_list in self.emergency_patterns.values():
            for pattern in patterns_list:
                matches = re.findall(pattern, text.lower())
                medical_terms.extend(matches)
        
        # Extract severity terms
        for patterns_list in self.severity_patterns.values():
            for pattern in patterns_list:
                matches = re.findall(pattern, text.lower())
                medical_terms.extend(matches)
        
        # Extract medical abbreviations (2-5 uppercase letters)
        abbreviations = re.findall(r'\b[A-Z]{2,5}\b', text)
        medical_terms.extend([abbr.lower() for abbr in abbreviations])
        
        # Extract important nouns (simple heuristic)
        important_words = re.findall(r'\b(?:blood|heart|lung|brain|bone|muscle|artery|vein|pressure|temperature|oxygen|pulse|breath|pain|injury|wound|symptom|treatment|medication|therapy|diagnosis|emergency|hospital|ambulance|doctor|nurse|patient)\b', text.lower())
        medical_terms.extend(important_words)
        
        # Remove duplicates and return
        return list(set(medical_terms))

def process_medical_documents(docs_directory: str, config: Dict[str, Any]) -> List[DocumentChunk]:
    """Process all medical documents in directory"""
    
    processor = MedicalDocumentProcessor(config)
    all_chunks = []
    
    docs_path = Path(docs_directory)
    if not docs_path.exists():
        print(f"⚠️  Documents directory not found: {docs_directory}")
        return []
    
    print(f"🔄 Processing medical documents from {docs_directory}...")
    
    # Process all .txt files
    txt_files = list(docs_path.glob("*.txt"))
    
    for file_path in txt_files:
        print(f"   Processing: {file_path.name}")
        try:
            chunks = processor.process_document(str(file_path))
            all_chunks.extend(chunks)
            print(f"     → {len(chunks)} chunks created")
        except Exception as e:
            print(f"     ❌ Error processing {file_path.name}: {e}")
    
    print(f"✅ Processed {len(txt_files)} files → {len(all_chunks)} total chunks")
    
    # Save processing statistics
    stats = _generate_processing_stats(all_chunks)
    print("\n📊 Processing Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return all_chunks

def _generate_processing_stats(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Generate statistics about processed chunks"""
    
    if not chunks:
        return {}
    
    stats = {
        'total_chunks': len(chunks),
        'average_words_per_chunk': sum(len(c.text.split()) for c in chunks) / len(chunks),
        'emergency_types': {},
        'severity_levels': {},
        'section_types': {}
    }
    
    # Count by categories
    for chunk in chunks:
        # Emergency types
        etype = chunk.emergency_type
        stats['emergency_types'][etype] = stats['emergency_types'].get(etype, 0) + 1
        
        # Severity levels
        severity = chunk.severity_level
        stats['severity_levels'][severity] = stats['severity_levels'].get(severity, 0) + 1
        
        # Section types
        section = chunk.section
        stats['section_types'][section] = stats['section_types'].get(section, 0) + 1
    
    return stats