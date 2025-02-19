import PyPDF2
import json
import spacy
from pathlib import Path

class VetDataExtractor:
    def __init__(self):
        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        
    def process_pdf(self, pdf_path, output_path):
        # Read PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Store processed data
            processed_data = []
            
            # Process each page
            for page in reader.pages:
                text = page.extract_text()
                # Process the extracted text
                processed_chunks = self.process_text(text)
                processed_data.extend(processed_chunks)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
            
        return processed_data

    def process_text(self, text):
        # Break text into meaningful chunks
        doc = self.nlp(text)
        
        chunks = []
        current_chunk = []
        
        for sent in doc.sents:
            current_chunk.append(sent.text)
            
            # Create a new chunk after collecting enough related sentences
            if self.is_chunk_complete(current_chunk):
                processed_chunk = self.create_qa_pair(current_chunk)
                if processed_chunk:
                    chunks.append(processed_chunk)
                current_chunk = []
        
        return chunks

    def create_qa_pair(self, sentences):
        text = ' '.join(sentences)
        
        # Skip if text is too short or doesn't contain meaningful content
        if len(text) < 50:
            return None
            
        # Create structured data
        data = {
            "source_text": text,
            "qa_pairs": self.generate_qa(text),
            "metadata": {
                "tags": self.extract_tags(text),
                "context": self.determine_context(text),
                "source_type": "textbook",
                "confidence": "high"
            }
        }
        
        return data

    def generate_qa(self, text):
        # Basic Q&A generation
        qa_pairs = []
        
        # Generate different types of questions based on content
        if any(keyword in text.lower() for keyword in ['symptom', 'sign']):
            qa_pairs.append({
                "question": "What are the symptoms or signs?",
                "answer": text
            })
            
        if any(keyword in text.lower() for keyword in ['treat', 'treatment', 'manage']):
            qa_pairs.append({
                "question": "What is the treatment or management approach?",
                "answer": text
            })
            
        return qa_pairs

    def extract_tags(self, text):
        # Extract relevant tags
        tags = set()
        
        # Add species tags
        if 'dog' in text.lower() or 'canine' in text.lower():
            tags.add('dogs')
        if 'cat' in text.lower() or 'feline' in text.lower():
            tags.add('cats')
            
        return list(tags)

# Usage example
if __name__ == "__main__":
    # Initialize extractor
    extractor = VetDataExtractor()
    
    # Set paths
    pdf_path = "data/raw/vet_book.pdf"
    output_path = "data/processed/processed_data.json"
    
    # Process PDF
    processed_data = extractor.process_pdf(pdf_path, output_path)
    print(f"Processed {len(processed_data)} chunks of information")