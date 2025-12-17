import httpx
import json
from typing import Optional, Dict, Any
import streamlit as st

class LLMClient:
    def __init__(self, gemini_key: str, hf_key: str):
        self.gemini_key = gemini_key
        self.hf_key = hf_key
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        self.hf_model = "microsoft/DialoGPT-medium"
    
    async def generate_content(
        self, 
        prompt: str, 
        action: str = "general",
        user: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Generate content with Gemini, fallback to HF"""
        
        # Try Gemini first
        result = await self._call_gemini(prompt)
        if result:
            return result
        
        # Fallback to Hugging Face
        result = await self._call_huggingface(prompt)
        if result:
            return result
        
        # Final fallback - mock response
        return self._mock_response(prompt)
    
    async def _call_gemini(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Gemini Flash API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gemini_url}?key={self.gemini_key}",
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
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
                        "model": "gemini-1.5-flash-latest",
                        "usage": {"total_tokens": 0}
                    }
        except Exception:
            pass  # Silent fail, will try fallback
        
        return None
    
    async def _call_huggingface(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Hugging Face Inference API"""
        try:
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
        except Exception:
            pass
        
        return None
    
    def _mock_response(self, prompt: str) -> Dict[str, Any]:
        """Mock response for demo when APIs are unavailable"""
        import hashlib
        seed = hashlib.md5(prompt.encode()).hexdigest()
        
        return {
            "text": json.dumps({
                "strengths": ["Strong brand", "Large user base"],
                "weaknesses": ["Limited AI features", "High pricing"],
                "market_position": "challenger",
                "threat_level": int(seed[0], 16) % 10 + 1,
                "opportunities": ["Mobile market", "Esports integration"]
            }, indent=2),
            "model": "mock_fallback",
            "usage": {"total_tokens": 0}
        }
