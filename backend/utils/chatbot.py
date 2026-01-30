import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class MedicalChatbot:
    """AI Chatbot for medical image diagnosis discussion"""
    
    def __init__(self, api_key=None):
        """
        Initialize chatbot with Google Gemini API
        
        Args:
            api_key: Google API key (if None, will try to get from .env file or environment)
        """
        # Try to get API key from: parameter > .env file > environment variable
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.chat = self.model.start_chat(history=[])
            self.enabled = True
        else:
            print("Warning: No API key provided. Chatbot will use rule-based responses.")
            self.enabled = False
        
        self.conversation_history = []
        
        # System prompt for medical context
        self.system_prompt = """You are a helpful medical AI assistant. Your role is to:
1. Explain medical imaging results in simple, understandable terms
2. Discuss diagnoses and their implications
3. Answer questions about medical conditions
4. Provide general health information

IMPORTANT:
- Always clarify that you're an AI assistant, not a doctor
- Recommend consulting healthcare professionals for medical decisions
- Be empathetic and supportive
- Use simple language to explain complex medical terms
- Ask clarifying questions when needed

Never:
- Provide specific treatment plans
- Make definitive diagnoses
- Replace professional medical advice
"""
    
    def chat_with_context(self, user_message: str, diagnosis_context: Dict = None, pdf_context: str = None):
        """
        Chat with diagnosis and PDF context
        
        Args:
            user_message: User's question
            diagnosis_context: Dict containing diagnosis results
            pdf_context: Extracted PDF text
        
        Returns:
            str: AI response
        """
        # Build context
        context_parts = [self.system_prompt]
        
        if diagnosis_context:
            context_parts.append("\n\n**Current Diagnosis Context:**")
            context_parts.append(f"- Image Type: {diagnosis_context.get('image_type', 'Unknown')}")
            context_parts.append(f"- Predicted Condition: {diagnosis_context.get('predicted_class', 'Unknown')}")
            context_parts.append(f"- Confidence: {diagnosis_context.get('confidence', 0):.1%}")
            
            if 'all_probabilities' in diagnosis_context:
                context_parts.append("\n**All Possibilities:**")
                for condition, prob in diagnosis_context['all_probabilities'].items():
                    if prob > 0.05:  # Only show >5% probability
                        context_parts.append(f"- {condition}: {prob:.1%}")
        
        if pdf_context:
            context_parts.append(f"\n\n**Medical Report Context:**\n{pdf_context[:1000]}")
        
        context_parts.append(f"\n\n**User Question:** {user_message}")
        
        full_prompt = "\n".join(context_parts)
        
        # Generate response
        if self.enabled:
            try:
                response = self.chat.send_message(full_prompt)
                ai_response = response.text
            except Exception as e:
                ai_response = self._fallback_response(user_message, diagnosis_context)
                print(f"API Error: {e}")
        else:
            ai_response = self._fallback_response(user_message, diagnosis_context)
        
        # Store in history
        self.conversation_history.append({
            'user': user_message,
            'assistant': ai_response
        })
        
        return ai_response
    
    def _fallback_response(self, user_message: str, diagnosis_context: Dict = None):
        """Rule-based fallback responses when API is unavailable"""
        
        user_lower = user_message.lower()
        
        # Explain diagnosis
        if any(word in user_lower for word in ['what', 'explain', 'mean', 'understand']):
            if diagnosis_context:
                predicted_class = diagnosis_context.get('predicted_class', 'Unknown')
                confidence = diagnosis_context.get('confidence', 0)
                
                return f"""Based on the analysis, the image shows signs of **{predicted_class}** with {confidence:.1%} confidence.

**What this means:**
This is an AI-based analysis of medical imaging. The model has detected patterns consistent with {predicted_class}.

**Important Note:**
- This is a preliminary AI analysis, not a definitive diagnosis
- Please consult with a qualified healthcare professional
- They will review the images along with your symptoms and medical history
- Further tests may be recommended for confirmation

Would you like me to explain more about this condition or the next steps?"""
        
        # Severity questions
        elif any(word in user_lower for word in ['serious', 'severe', 'dangerous', 'worry']):
            return """I understand your concern. The severity depends on several factors:

1. **Stage/Grade**: Early detection often means better outcomes
2. **Symptoms**: What symptoms are you experiencing?
3. **Medical History**: Previous conditions matter
4. **Professional Opinion**: A doctor's assessment is crucial

**Next Steps:**
- Schedule an appointment with a specialist
- Bring all your medical reports
- Discuss symptoms in detail
- Ask about treatment options

Remember, many conditions are manageable with proper care. Don't hesitate to seek professional help."""
        
        # Treatment questions
        elif any(word in user_lower for word in ['treatment', 'cure', 'medicine', 'therapy']):
            return """Treatment options vary greatly depending on the specific condition, its severity, and your overall health.

**General Approach:**
1. **Diagnosis Confirmation**: Ensure accurate diagnosis
2. **Treatment Plan**: Developed by your healthcare team
3. **Options May Include**:
   - Medications
   - Lifestyle changes
   - Physical therapy
   - Surgical intervention (if needed)
   - Regular monitoring

**Important:**
I cannot recommend specific treatments. Please discuss with your doctor who can:
- Review your complete medical history
- Consider your specific situation
- Recommend personalized treatment
- Monitor your progress

Is there anything else you'd like to know?"""
        
        # Prognosis/outcome
        elif any(word in user_lower for word in ['prognosis', 'outcome', 'recover', 'better']):
            return """Outcomes vary significantly based on individual factors:

**Positive Factors:**
- Early detection
- Following treatment plan
- Healthy lifestyle
- Regular check-ups

**What You Can Do:**
- Stay informed about your condition
- Follow medical advice
- Maintain communication with healthcare team
- Take prescribed medications
- Attend follow-up appointments

Many conditions have excellent outcomes with proper management. Your healthcare provider can give you specific information based on your case.

Do you have specific concerns about your situation?"""
        
        # General question
        else:
            return """I'm here to help you understand medical imaging results and health information.

I can help with:
- Explaining what the diagnosis might mean
- Discussing general information about conditions
- Suggesting questions to ask your doctor
- Understanding medical terms

However, I cannot:
- Provide definitive diagnoses
- Recommend specific treatments
- Replace professional medical advice

**Please consult a healthcare professional for:**
- Accurate diagnosis
- Treatment recommendations
- Medical decisions

What specific information would you like to know?"""
    
    def get_conversation_history(self):
        """Get full conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        if self.enabled:
            self.chat = self.model.start_chat(history=[])


if __name__ == "__main__":
    print("Medical Chatbot initialized")
    print("\nFeatures:")
    print("- Context-aware responses using diagnosis results")
    print("- PDF report integration")
    print("- Conversation history tracking")
    print("- Fallback rule-based responses")
    print("\nNote: Set GOOGLE_API_KEY environment variable to use Gemini API")
