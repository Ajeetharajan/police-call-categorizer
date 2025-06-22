import re
import spacy
import pandas as pd
from datetime import datetime
import json
from src.config.config import CRIME_CATEGORIES # ADD THIS LINE

class CrimeAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.crime_categories = CRIME_CATEGORIES # CHANGE THIS LINE to use the imported categories
        # ... rest of your __init__ remains the same ...
        
    def categorize_complaint(self, text):
        """Categorize the complaint based on keywords"""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.crime_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score
        
        if scores:
            primary_category = max(scores, key=scores.get)
            confidence = scores[primary_category] / len(self.crime_categories[primary_category])
            return primary_category, confidence
        
        return "unknown", 0.0
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = {
            'persons': [],
            'locations': [],
            'organizations': [],
            'dates': [],
            'money': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON']:
                entities['persons'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                entities['locations'].append(ent.text)
            elif ent.label_ in ['ORG']:
                entities['organizations'].append(ent.text)
            elif ent.label_ in ['DATE', 'TIME']:
                entities['dates'].append(ent.text)
            elif ent.label_ in ['MONEY']:
                entities['money'].append(ent.text)
        
        return entities
    
    def extract_urgency_level(self, text):
        """Determine urgency level based on keywords"""
        urgent_keywords = ['emergency', 'urgent', 'immediate', 'help', 'danger', 'weapon']
        moderate_keywords = ['soon', 'quick', 'fast']
        
        text_lower = text.lower()
        urgent_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        moderate_count = sum(1 for keyword in moderate_keywords if keyword in text_lower)
        
        if urgent_count > 0:
            return "HIGH"
        elif moderate_count > 0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def extract_contact_info(self, text):
        """Extract phone numbers and addresses"""
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        
        # Simple address pattern (can be improved)
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)'
        addresses = re.findall(address_pattern, text, re.IGNORECASE)
        
        return {
            'phone_numbers': phones,
            'addresses': addresses
        }
    
    def analyze(self, text):
        """Complete analysis of the complaint text"""
        category, confidence = self.categorize_complaint(text)
        entities = self.extract_entities(text)
        urgency = self.extract_urgency_level(text)
        contact_info = self.extract_contact_info(text)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'complaint_category': category,
            'confidence_score': round(confidence, 2),
            'urgency_level': urgency,
            'extracted_entities': entities,
            'contact_information': contact_info,
            'text_length': len(text),
            'word_count': len(text.split())
        }
