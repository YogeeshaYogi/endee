"""Answer generation using Google Gemini API."""

import google.generativeai as genai
from typing import List
import logging
from config import Config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generates answers using Google Gemini language models."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model or Config.GEMINI_MODEL
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        else:
            self.model = None
            logger.warning("Gemini API key not provided. Answer generation will not work.")
            raise Exception("Gemini API key not configured")
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate an answer based on the question and retrieved contexts."""
        if not self.model:
            return "Gemini API key not configured. Cannot generate answers."
        
        try:
            # Combine contexts
            combined_context = "\n\n".join(contexts)
            
            # Create prompt for Gemini
            prompt = self._create_prompt(question, combined_context)
            
            # Configure generation parameters
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 300,
            }
            
            # Generate answer using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                answer = response.text.strip()
                logger.info("Answer generated successfully with Gemini")
                return answer
            else:
                return "❌ Gemini did not generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            if "quota" in str(e).lower() or "billing" in str(e).lower():
                return f"💳 Gemini API quota exceeded. Please check your usage at https://console.cloud.google.com/"
            elif "api_key" in str(e).lower():
                return f"🔑 Invalid Gemini API key. Get one at https://makersuite.google.com/app/apikey"
            else:
                return f"❌ Gemini API error: {str(e)}"
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the language model."""
        return f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question completely, please state what information is missing."""

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text."""
        if not self.model:
            return "Summary generation requires Gemini API key."
        
        try:
            prompt = f"Please summarize the following text in {max_length} words or less:\n\n{text}"
            
            generation_config = {
                "temperature": 0.3,
                "max_output_tokens": max_length * 2,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "Summary generation failed: No response generated"
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Summary generation failed: {str(e)}"