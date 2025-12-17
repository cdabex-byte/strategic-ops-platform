import os
import httpx
import json
from typing import Optional, Dict, Any

class LLMClient:
    def __init__(self, gemini_key: str, hf_key: str):
        self.gemini_key = gemini_key
        self.hf_key = hf_key
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-latest:generateContent"
        self.hf_model = "microsoft/DialoGPT-medium"  # Free, fallback model
    
    async def generate_content(
        self, 
        prompt: str, 
        action: str = "general",
        user: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Generate content with Gemini, fallback to HF"""
        
        # Try Gemini first (primary)
        try:
            result = await self._call_gemini(prompt)
            if result:
                # Log success
                from .sheets_logger import logger
                if logger:
                    logger.log_interaction(
                        user=user,
                        action=action,
                        input_data={"prompt": prompt},
                        output_data=result,
                        model="gemini-2.5-flash-latest",
                        tokens=result.get('usage', {}).get('total_tokens', 0),
                        cost_estimate=0.0,  # Free tier
                        status="success"
                    )
                return result
        
        except Exception as e:
            print(f"Gemini failed: {e}, trying Hugging Face...")
            
            # Log Gemini failure
            from .sheets_logger import logger
            if logger:
                logger.log_interaction(
                    user=user,
                    action=f"{action}_gemini_failed",
                    input_data={"error": str(e)},
                    output_data={},
                    model="gemini-2.5-flash-latest",
                    status="failed"
                )
        
        # Fallback to Hugging Face
        try:
            result = await self._call_huggingface(prompt)
            if result:
                # Log HF success
                from .sheets_logger import logger
                if logger:
                    logger.log_interaction(
                        user=user,
                        action=action,
                        input_data={"prompt": prompt},
                        output_data=result,
                        model=self.hf_model,
                        tokens=len(prompt.split()) * 2,
                        cost_estimate=0.0,
                        status="success_hf_fallback"
                    )
                return result
        
        except Exception as e:
            print(f"Hugging Face also failed: {e}")
            
            # Log total failure
            from .sheets_logger import logger
            if logger:
                logger.log_interaction(
                    user=user,
                    action=f"{action}_total_failure",
                    input_data={"prompt": prompt},
                    output_data={"error": str(e)},
                    model="none",
                    status="failed"
                )
        
        return None
    
    async def _call_gemini(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Gemini Flash Latest API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.gemini_url + f"?key={self.gemini_key}",
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 2048,
                        "topP": 0.95
                    }
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data['candidates'][0]['content']['parts'][0]['text'],
                    "model": "gemini-2.5-flash-latest",
                    "usage": {"total_tokens": 0}  # Gemini free tier doesn't return tokens
                }
            return None
    
    async def _call_huggingface(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Hugging Face Inference API (free tier)"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{self.hf_model}",
                headers={"Authorization": f"Bearer {self.hf_key}"},
                json={"inputs": prompt, "parameters": {"max_length": 500}},
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data[0]['generated_text'],
                    "model": self.hf_model,
                    "usage": {"total_tokens": len(prompt.split()) * 2}
                }
            return None
