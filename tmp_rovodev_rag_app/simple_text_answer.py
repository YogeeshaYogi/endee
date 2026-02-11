"""Simple text-based answer generation without AI models."""

from typing import List
import re

class SimpleTextAnswerer:
    """Generate simple answers by extracting relevant text from context."""
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """Generate a simple answer by extracting relevant sentences."""
        try:
            # Combine all contexts
            combined_text = " ".join(contexts)
            
            # Extract key terms from question
            question_words = self._extract_keywords(question.lower())
            
            # Split context into sentences
            sentences = re.split(r'[.!?]+', combined_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            # Score sentences based on keyword matches
            scored_sentences = []
            for sentence in sentences:
                score = self._score_sentence(sentence.lower(), question_words)
                if score > 0:
                    scored_sentences.append((score, sentence.strip()))
            
            # Sort by score and take top sentences
            scored_sentences.sort(reverse=True, key=lambda x: x[0])
            top_sentences = [sent[1] for sent in scored_sentences[:3]]
            
            if top_sentences:
                answer = "Based on the documents, " + " ".join(top_sentences)
                if len(answer) > 500:
                    answer = answer[:500] + "..."
                return answer
            else:
                return "I found relevant documents but couldn't extract a specific answer to your question. Please review the source documents for more details."
                
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from the question."""
        # Remove common question words
        stop_words = {'what', 'is', 'are', 'how', 'why', 'where', 'when', 'who', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Simple word extraction
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _score_sentence(self, sentence: str, keywords: List[str]) -> int:
        """Score a sentence based on keyword matches."""
        score = 0
        for keyword in keywords:
            if keyword in sentence:
                score += 1
                # Bonus for exact word match
                if f" {keyword} " in f" {sentence} ":
                    score += 1
        return score