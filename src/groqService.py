from groq import Groq
from typing import Any, List, Optional, Dict
from src.config.settings import (
    GROQ_API_KEY,
    GROQ_LLM_MODEL,
    GROQ_MAX_COMPLETION_TOKENS,
    GROQ_TEMPERATURE,
    GROQ_STREAM,
)

class GroqLLMWrapper:
    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = GROQ_LLM_MODEL,
        max_completion_tokens: int = GROQ_MAX_COMPLETION_TOKENS,
        temperature: float = GROQ_TEMPERATURE,
        stream: bool = GROQ_STREAM,
    ):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.stream = stream

    def send_prompt(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Any:
        messages = context if context else []
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=top_p,
            stream=self.stream,
            stop=stop,
        )
        
        return completion

    def send_prompt_json(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> Any:
        """Send prompt with JSON response format enforced"""
        messages = context if context else []
        messages.append({"role": "user", "content": prompt})

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=top_p,
            stream=False,  # JSON mode requires stream=False
            response_format={"type": "json_object"},
            stop=stop,
        )
        
        return completion