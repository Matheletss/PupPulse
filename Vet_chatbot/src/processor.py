from typing import List, Dict, Optional
import json
import re
from pathlib import Path
import pandas as pd
from collections import defaultdict

class VetDataProcessor:
    def __init__(self):
        self.categories = {
            'diseases': ['symptom', 'disease', 'condition', 'infection'],
            'behavior': ['behavior', 'training', 'anxiety', 'stress'],
            'emergency': ['emergency', 'urgent', 'immediate', 'critical'],
            'preventive': ['prevent', 'vaccination', 'nutrition', 'diet'],
            'breeds': ['breed', 'purebred', 'mixed breed']
        }
        
    def process_raw_data(self, input_path: str, output_path: str):
        """
        Process raw JSON data into refined training data
        """
        # Load raw data
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        processed_data = []
        
        for entry in raw_data:
            # Enhance Q&A pairs
            enhanced_qa = self.enhance_qa_pairs(entry['qa_pairs'])
            
            # Categorize content
            category = self.categorize_content(entry['source_text'])
            
            # Create structured entry
            processed_entry = {
                'original_text': entry['source_text'],
                'qa_pairs': enhanced_qa,
                'category': category,
                'metadata': self.enhance_metadata(entry['metadata'])
            }
            
            processed_data.append(processed_entry)
            
        # Save processed data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
            
        return processed_data
    
    def enhance_qa_pairs(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Enhance Q&A pairs with additional context and variations
        """
        enhanced_pairs = []
        
        for qa in qa_pairs:
            # Original Q&A
            enhanced_pairs.append(qa)
            
            # Generate variations
            variations = self.generate_question_variations(qa['question'], qa['answer'])
            enhanced_pairs.extend(variations)
            
            # Add follow-up questions if relevant
            followups = self.generate_followup_questions(qa['question'], qa['answer'])
            enhanced_pairs.extend(followups)
            
        return enhanced_pairs
    
    def generate_question_variations(self, question: str, answer: str) -> List[Dict]:
        """
        Generate variations of the same question
        """
        variations = []
        
        # Example: Convert "What are the symptoms?" to different forms
        if 'symptoms' in question.lower():
            variations.extend([
                {
                    'question': 'What signs should I look out for?',
                    'answer': answer
                },
                {
                    'question': 'How can I tell if my pet has this condition?',
                    'answer': answer
                }
            ])
            
        return variations
    
    def generate_followup_questions(self, question: str, answer: str) -> List[Dict]:
        """
        Generate relevant follow-up questions based on the answer
        """
        followups = []
        
        # Check if answer mentions treatments
        if any(word in answer.lower() for word in ['treat', 'medication', 'therapy']):
            followups.append({
                'question': 'What are the treatment options?',
                'answer': self.extract_treatment_info(answer)
            })
            
        # Check if answer mentions prevention
        if any(word in answer.lower() for word in ['prevent', 'avoid', 'reduce risk']):
            followups.append({
                'question': 'How can this be prevented?',
                'answer': self.extract_prevention_info(answer)
            })
            
        return followups
    
    def categorize_content(self, text: str) -> str:
        """
        Categorize content based on keywords and context
        """
        text_lower = text.lower()
        
        for category, keywords in self.categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
                
        return 'general'
    
    def enhance_metadata(self, metadata: Dict) -> Dict:
        """
        Enhance metadata with additional useful information
        """
        enhanced = metadata.copy()
        
        # Add complexity level
        enhanced['complexity'] = self.determine_complexity(metadata.get('tags', []))
        
        # Add content type
        enhanced['content_type'] = self.determine_content_type(metadata.get('tags', []))
        
        return enhanced
    
    def determine_complexity(self, tags: List[str]) -> str:
        """
        Determine content complexity based on tags
        """
        emergency_keywords = {'emergency', 'critical', 'urgent'}
        technical_keywords = {'medical', 'clinical', 'diagnostic'}
        
        if any(tag in emergency_keywords for tag in tags):
            return 'high'
        elif any(tag in technical_keywords for tag in tags):
            return 'medium'
        return 'basic'
    
    def determine_content_type(self, tags: List[str]) -> str:
        """
        Determine type of content based on tags
        """
        if 'emergency' in tags:
            return 'emergency'
        elif 'medical' in tags:
            return 'medical'
        elif 'behavior' in tags:
            return 'behavioral'
        return 'general'
    
    def extract_treatment_info(self, text: str) -> str:
        """
        Extract treatment-related information from text
        """
        # Find sentences containing treatment information
        sentences = text.split('.')
        treatment_sentences = [s for s in sentences if any(word in s.lower() 
                             for word in ['treat', 'medication', 'therapy'])]
        
        return '. '.join(treatment_sentences) + '.' if treatment_sentences else ''
    
    def extract_prevention_info(self, text: str) -> str:
        """
        Extract prevention-related information from text
        """
        sentences = text.split('.')
        prevention_sentences = [s for s in sentences if any(word in s.lower() 
                              for word in ['prevent', 'avoid', 'reduce'])]
        
        return '. '.join(prevention_sentences) + '.' if prevention_sentences else ''

    def generate_training_data(self, processed_data: List[Dict], output_path: str):
        """
        Generate final training data format
        """
        training_data = []
        
        for entry in processed_data:
            for qa in entry['qa_pairs']:
                training_example = {
                    'input': qa['question'],
                    'output': qa['answer'],
                    'category': entry['category'],
                    'metadata': entry['metadata']
                }
                training_data.append(training_example)
                
        # Save training data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=4, ensure_ascii=False)
            
        return training_data
    
    def categorize_content(self, text: str) -> Dict:
    categories = {
        'breed': [],
        'diseases': [],
        'predisposition_level': None
    }
    
    # Extract breed information
    breed_patterns = [
        r'in ([A-Z][a-z]+ [A-Z][a-z]+)',  # For breed names like "German Shepherd"
        r'in ([A-Z][a-z]+)'                # For single word breeds like "Poodle"
    ]
    
    for pattern in breed_patterns:
        matches = re.findall(pattern, text)
        if matches:
            categories['breed'].extend(matches)
    
    # Extract disease information
    disease_patterns = [
        r'([\w\s]+) is (?:more|less) common',
        r'predisposed to ([\w\s]+)',
        r'risk of ([\w\s]+)'
    ]
    
    for pattern in disease_patterns:
        matches = re.findall(pattern, text)
        if matches:
            categories['diseases'].extend(matches)
    
    return categories

# Usage example
if __name__ == "__main__":
    processor = VetDataProcessor()
    
    # Process raw data
    raw_data_path = "data/processed/raw_data.json"
    processed_data_path = "data/processed/processed_data.json"
    training_data_path = "data/processed/training_data.json"
    
    # Process the data
    processed_data = processor.process_raw_data(raw_data_path, processed_data_path)
    
    # Generate training data
    training_data = processor.generate_training_data(processed_data, training_data_path)
    
    print(f"Generated {len(training_data)} training examples")