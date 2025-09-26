"""
Ollama/LLaMA Local Inference Service
Constitutional compliance: local-first operation, privacy-preserving
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
import httpx
from ollama import AsyncClient

from core.config import settings

logger = logging.getLogger(__name__)

class OllamaService:
    """Ollama service for local LLM inference with constitutional compliance"""
    
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.available_models: List[str] = []
        self.current_model: str = settings.ollama_model
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to Ollama service"""
        try:
            self.client = AsyncClient(host=settings.ollama_host)
            
            # Test connection
            await self.client.list()
            
            # Get available models
            models_response = await self.client.list()
            self.available_models = [model['name'] for model in models_response['models']]
            
            # Verify current model is available
            if self.current_model not in self.available_models:
                logger.warning(f"Model {self.current_model} not available. Available: {self.available_models}")
                if self.available_models:
                    self.current_model = self.available_models[0]
                    logger.info(f"Switched to model: {self.current_model}")
            
            self._connected = True
            logger.info(f"Connected to Ollama at {settings.ollama_host}")
            logger.info(f"Available models: {self.available_models}")
            logger.info(f"Current model: {self.current_model}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Ollama service"""
        if self.client:
            await self.client.close()
            self.client = None
        self._connected = False
        logger.info("Disconnected from Ollama")
    
    async def generate_text(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        mock_data: bool = True
    ) -> Dict[str, Any]:
        """Generate text using local LLM with constitutional compliance"""
        
        if not self._connected or not self.client:
            raise RuntimeError("Ollama service not connected")
        
        if mock_data:
            logger.info("Using mock data for text generation")
            return await self._mock_generate_text(prompt, model or self.current_model)
        
        try:
            model_to_use = model or self.current_model
            
            # Prepare request
            request_data = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                }
            }
            
            if system_prompt:
                request_data["system"] = system_prompt
            
            if max_tokens:
                request_data["options"]["num_predict"] = max_tokens
            
            # Generate response
            response = await self.client.generate(**request_data)
            
            return {
                "text": response["response"],
                "model": model_to_use,
                "tokens_used": response.get("eval_count", 0),
                "generation_time": response.get("total_duration", 0) / 1e9,  # Convert to seconds
                "mock_data": False
            }
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def extract_concepts(self, text: str, mock_data: bool = True) -> List[str]:
        """Extract concepts from text using local LLM"""
        
        system_prompt = """You are a concept extraction specialist. Extract key concepts, topics, and themes from the given text. 
        Return only the concepts as a comma-separated list, without explanations or additional text."""
        
        prompt = f"Extract key concepts from this text:\n\n{text[:2000]}"  # Limit input length
        
        try:
            response = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent extraction
                mock_data=mock_data
            )
            
            # Parse concepts from response
            concepts_text = response["text"].strip()
            concepts = [concept.strip() for concept in concepts_text.split(",")]
            concepts = [concept for concept in concepts if concept]  # Remove empty strings
            
            return concepts[:20]  # Limit to 20 concepts
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []
    
    async def generate_summary(self, text: str, mock_data: bool = True) -> str:
        """Generate summary using local LLM"""
        
        system_prompt = """You are a summarization specialist. Create a concise, accurate summary of the given text. 
        Focus on key points, main arguments, and important findings. Maintain scientific objectivity."""
        
        prompt = f"Summarize this text:\n\n{text[:3000]}"  # Limit input length
        
        try:
            response = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
                mock_data=mock_data
            )
            
            return response["text"].strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary generation failed"
    
    async def evaluate_content(
        self, 
        content: str, 
        evaluation_type: str,
        mock_data: bool = True
    ) -> Dict[str, str]:
        """Generate evaluative feedback using constitutional framework"""
        
        system_prompt = """You are an evaluative feedback specialist. Analyze the given content and provide feedback 
        answering these four constitutional questions:
        1. What's good? (positive aspects)
        2. What's broken? (problems or issues)
        3. What works but shouldn't? (functioning but inappropriate)
        4. What doesn't but pretends to? (non-functional but claims to work)
        
        Be specific, objective, and constructive."""
        
        prompt = f"Evaluate this {evaluation_type} content:\n\n{content[:2000]}"
        
        try:
            response = await self.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.5,
                mock_data=mock_data
            )
            
            # Parse the four-part evaluation
            evaluation_text = response["text"]
            
            # Simple parsing - in production, use more sophisticated parsing
            lines = evaluation_text.split('\n')
            evaluation = {
                "whats_good": "Analysis pending",
                "whats_broken": "Analysis pending", 
                "what_works_but_shouldnt": "Analysis pending",
                "what_doesnt_but_pretends_to": "Analysis pending"
            }
            
            # Try to extract structured feedback
            for line in lines:
                line = line.strip()
                if line.lower().startswith("what's good"):
                    evaluation["whats_good"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif line.lower().startswith("what's broken"):
                    evaluation["whats_broken"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif line.lower().startswith("what works but shouldn't"):
                    evaluation["what_works_but_shouldnt"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif line.lower().startswith("what doesn't but pretends"):
                    evaluation["what_doesnt_but_pretends_to"] = line.split(":", 1)[1].strip() if ":" in line else line
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Content evaluation failed: {e}")
            return {
                "whats_good": "Evaluation failed",
                "whats_broken": "Evaluation failed",
                "what_works_but_shouldnt": "Evaluation failed", 
                "what_doesnt_but_pretends_to": "Evaluation failed"
            }
    
    async def _mock_generate_text(self, prompt: str, model: str) -> Dict[str, Any]:
        """Mock text generation for development"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        return {
            "text": f"Mock response for prompt: {prompt[:50]}...",
            "model": model,
            "tokens_used": 25,
            "generation_time": 0.5,
            "mock_data": True
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health"""
        if not self._connected:
            return {
                "status": "not_connected",
                "available_models": [],
                "current_model": None,
                "mock_data": True
            }
        
        try:
            # Test with a simple request
            await self.client.list()
            
            return {
                "status": "healthy",
                "available_models": self.available_models,
                "current_model": self.current_model,
                "mock_data": False
            }
            
        except Exception as e:
            return {
                "status": f"error: {str(e)}",
                "available_models": self.available_models,
                "current_model": self.current_model,
                "mock_data": True
            }
    
    @property
    def is_connected(self) -> bool:
        """Check if Ollama service is connected"""
        return self._connected

# Global Ollama service instance
ollama_service = OllamaService()
