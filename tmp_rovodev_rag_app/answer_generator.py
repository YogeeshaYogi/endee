"""Answer generation using OpenAI API."""

from openai import OpenAI
from typing import List
import logging
from config import Config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Generates answers using OpenAI's language models."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not provided. Answer generation will not work.")
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate an answer based on the question and retrieved contexts."""
        if not self.client:
            return "OpenAI API key not configured. Cannot generate answers."
        
        try:
            # Combine contexts
            combined_context = "\n\n".join(contexts)
            
            # Create prompt
            prompt = self._create_prompt(question, combined_context)
            
            # Generate answer using OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. "
                                 "If the context doesn't contain enough information to answer the question, "
                                 "say so clearly. Always cite information from the context when possible."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.1,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"I encountered an error while generating an answer: {str(e)}"
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the language model."""
        return f"""Context:
{context}

Question: {question}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question completely, please state what information is missing."""

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text."""
        if not self.client:
            return "Summary generation requires OpenAI API key."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please summarize the following text in {max_length} words or less:\n\n{text}"
                    }
                ],
                max_tokens=max_length * 2,  # Rough estimate for tokens
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Summary generation failed: {str(e)}"