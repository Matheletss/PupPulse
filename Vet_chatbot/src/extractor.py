import PyPDF2
import json
import spacy
from pathlib import Path
from typing import List, Dict, Optional
import logging

class VetDataExtractor:
    """
    A class for extracting and processing veterinary information from PDF documents.
    Converts textbook content into structured Q&A format suitable for chatbot training.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the extractor with spaCy model and configure logging.
        
        Args:
            model_name: Name of the spaCy model to use for text processing
        """
        # Initialize spaCy with more comprehensive model
        self.nlp = spacy.load(model_name)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define common veterinary terms for better content recognition
        self.symptoms_keywords = {
            'symptom', 'sign', 'clinical presentation', 'manifestation',
            'indication', 'presenting complaint'
        }
        
        self.treatment_keywords = {
            'treat', 'treatment', 'manage', 'therapy', 'intervention',
            'prescription', 'medication', 'remedy'
        }
        
        self.species_mapping = {
            'dog': ['canine', 'puppy', 'dog'],
            'cat': ['feline', 'kitten', 'cat'],
            'bird': ['avian', 'parrot', 'parakeet'],
            'rabbit': ['bunny', 'rabbit', 'hare'],
            'hamster': ['hamster', 'gerbil', 'rodent']
        }

    def process_pdf(self, pdf_path: str, output_path: str) -> List[Dict]:
        """
        Process a veterinary PDF document and extract structured information.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path where processed JSON should be saved
        
        Returns:
            List of processed data chunks
        """
        self.logger.info(f"Starting to process PDF: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                processed_data = []
                
                for page_num, page in enumerate(reader.pages, 1):
                    self.logger.info(f"Processing page {page_num}/{len(reader.pages)}")
                    text = page.extract_text()
                    
                    # Skip pages with minimal content
                    if len(text.strip()) < 100:
                        self.logger.warning(f"Page {page_num} contains minimal text, skipping")
                        continue
                        
                    processed_chunks = self.process_text(text)
                    processed_data.extend(processed_chunks)
                    
            # Save processed data
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
                
            self.logger.info(f"Successfully processed {len(processed_data)} chunks")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise

    def process_text(self, text: str) -> List[Dict]:
        """
        Process text into meaningful chunks for Q&A generation.
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            List of processed chunks with Q&A pairs
        """
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        
        for sent in doc.sents:
            # Skip sentences that are too short or contain only numbers/special characters
            if len(sent.text.strip()) < 10 or not any(c.isalpha() for c in sent.text):
                continue
                
            current_chunk.append(sent.text)
            
            # Create a new chunk after collecting related sentences
            if self._is_chunk_complete(current_chunk):
                processed_chunk = self.create_qa_pair(current_chunk)
                if processed_chunk:
                    chunks.append(processed_chunk)
                current_chunk = []
        
        return chunks

    def create_qa_pair(self, sentences: List[str]) -> Optional[Dict]:
        """
        Create structured Q&A pairs from a chunk of text.
        
        Args:
            sentences: List of related sentences
            
        Returns:
            Dictionary containing source text, Q&A pairs, and metadata
        """
        text = ' '.join(sentences)
        
        # Skip if text is too short or doesn't contain meaningful content
        if len(text) < 50:
            return None
            
        qa_pairs = self.generate_qa(text)
        if not qa_pairs:
            return None
            
        data = {
            "source_text": text,
            "qa_pairs": qa_pairs,
            "metadata": {
                "tags": self.extract_tags(text),
                "context": self.determine_context(text),
                "source_type": "textbook",
                "confidence": self._calculate_confidence(text)
            }
        }
        
        return data

    def generate_qa(self, text: str) -> List[Dict]:
        """
        Generate relevant Q&A pairs based on content analysis.
        
        Args:
            text: Processed text chunk
            
        Returns:
            List of Q&A pairs
        """
        qa_pairs = []
        text_lower = text.lower()
        
        # Generate symptom-related questions
        if any(keyword in text_lower for keyword in self.symptoms_keywords):
            qa_pairs.append({
                "question": "What are the symptoms or signs that might indicate this condition?",
                "answer": text
            })
        
        # Generate treatment-related questions
        if any(keyword in text_lower for keyword in self.treatment_keywords):
            qa_pairs.append({
                "question": "What are the recommended treatment approaches or management strategies?",
                "answer": text
            })
        
        # Generate prevention-related questions
        if 'prevent' in text_lower or 'prevention' in text_lower:
            qa_pairs.append({
                "question": "What preventive measures or precautions should be taken?",
                "answer": text
            })
        
        return qa_pairs

    def extract_tags(self, text: str) -> List[str]:
        """
        Extract relevant tags including species, conditions, and topics.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of relevant tags
        """
        tags = set()
        text_lower = text.lower()
        
        # Add species tags
        for species, keywords in self.species_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.add(species)
        
        # Add condition tags
        if any(keyword in text_lower for keyword in self.symptoms_keywords):
            tags.add('symptoms')
        if any(keyword in text_lower for keyword in self.treatment_keywords):
            tags.add('treatment')
        
        return list(tags)

    def _is_chunk_complete(self, chunk: List[str]) -> bool:
        """
        Determine if the current chunk of sentences forms a complete thought.
        
        Args:
            chunk: List of sentences
            
        Returns:
            Boolean indicating if chunk is complete
        """
        text = ' '.join(chunk)
        
        # Check if chunk is getting too long
        if len(text) > 1000:
            return True
            
        # Check for natural breaks in content
        if text.strip().endswith(('.', '!', '?')):
            return True
            
        return False

    def _calculate_confidence(self, text: str) -> str:
        """
        Calculate confidence score for the extracted information.
        
        Args:
            text: Processed text
            
        Returns:
            Confidence level as string
        """
        # Simple heuristic based on text length and keyword presence
        if len(text) < 100:
            return "low"
        elif len(text) > 500 and any(keyword in text.lower() for keyword in 
                                   self.symptoms_keywords | self.treatment_keywords):
            return "high"
        return "medium"

if __name__ == "__main__":
    # Initialize extractor with logging
    extractor = VetDataExtractor()
    
    # Set paths
    pdf_path = "data/raw/vet_book.pdf"
    output_path = "data/processed/processed_data.json"
    
    # Process PDF
    try:
        processed_data = extractor.process_pdf(pdf_path, output_path)
        print(f"Successfully processed {len(processed_data)} chunks of information")
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")