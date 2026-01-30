from PyPDF2 import PdfReader
import re
from pathlib import Path


class PDFExtractor:
    """Extract text from PDF medical reports"""
    
    def __init__(self):
        self.medical_keywords = [
            'diagnosis', 'symptoms', 'findings', 'impression', 'conclusion',
            'history', 'examination', 'test', 'result', 'treatment', 'medication',
            'patient', 'doctor', 'hospital', 'clinic', 'laboratory'
        ]
    
    def extract_text(self, pdf_path):
        """Extract all text from PDF"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def extract_structured_info(self, pdf_path):
        """Extract structured information from medical report"""
        text = self.extract_text(pdf_path)
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        # Extract sections
        sections = {
            'full_text': text,
            'diagnosis': self._extract_section(text, ['diagnosis', 'impression']),
            'findings': self._extract_section(text, ['findings', 'observations']),
            'symptoms': self._extract_section(text, ['symptoms', 'complaints', 'history']),
            'medications': self._extract_section(text, ['medication', 'drugs', 'prescription']),
            'recommendations': self._extract_section(text, ['recommendation', 'advice', 'follow-up'])
        }
        
        return sections
    
    def _extract_section(self, text, keywords):
        """Extract section based on keywords"""
        text_lower = text.lower()
        
        for keyword in keywords:
            # Find keyword position
            pattern = rf'{keyword}[:\-\s]+(.*?)(?=\n\n|\n[A-Z]{{3,}}|$)'
            match = re.search(pattern, text_lower, re.DOTALL)
            
            if match:
                # Get the actual text (not lowercased)
                start_pos = match.start(1)
                end_pos = match.end(1)
                section_text = text[start_pos:end_pos].strip()
                
                if len(section_text) > 10:  # Minimum length
                    return section_text
        
        return "Not found"
    
    def is_medical_report(self, pdf_path):
        """Check if PDF is likely a medical report"""
        text = self.extract_text(pdf_path).lower()
        
        # Count medical keywords
        keyword_count = sum(1 for keyword in self.medical_keywords if keyword in text)
        
        # If at least 3 medical keywords found, likely a medical report
        return keyword_count >= 3
    
    def summarize_key_points(self, pdf_path):
        """Extract key points from medical report"""
        sections = self.extract_structured_info(pdf_path)
        
        key_points = []
        
        if sections['diagnosis'] != "Not found":
            key_points.append(f"**Diagnosis:** {sections['diagnosis'][:200]}")
        
        if sections['findings'] != "Not found":
            key_points.append(f"**Key Findings:** {sections['findings'][:200]}")
        
        if sections['symptoms'] != "Not found":
            key_points.append(f"**Symptoms:** {sections['symptoms'][:200]}")
        
        if sections['medications'] != "Not found":
            key_points.append(f"**Medications:** {sections['medications'][:200]}")
        
        return key_points if key_points else ["Unable to extract structured information from this PDF"]


if __name__ == "__main__":
    # Test PDF extractor
    print("PDF Extractor initialized successfully")
    
    # Example usage
    extractor = PDFExtractor()
    print("\nSupported features:")
    print("- Extract full text from PDF")
    print("- Extract structured sections (diagnosis, findings, symptoms, etc.)")
    print("- Identify medical reports")
    print("- Summarize key points")
