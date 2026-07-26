"""
OpenAI-compatible API client for calibration probes.

Endpoint-agnostic — works with any OpenAI-compatible API:
- Direct providers (OpenAI, Anthropic via proxy)
- Self-hosted (vLLM, Ollama, llama.cpp)
- Gateways (LiteLLM, OpenRouter)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class CalibrationParams:
    """Mandatory parameters for reproducible calibration."""
    temperature: float = 0
    top_p: float = 1.0
    seed: int | None = 42
    max_tokens: int = 4096


class CalibrationClient:
    """
    OpenAI-compatible client for calibration probes.
    
    Enforces reproducibility parameters (temp=0, seed=42).
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        params: CalibrationParams | None = None,
    ):
        """
        Initialize calibration client.
        
        Args:
            base_url: API base URL (e.g., "http://localhost:11434/v1")
            api_key: API key (use "ollama" for Ollama)
            timeout: Request timeout in seconds
            params: Calibration parameters (defaults to temp=0, seed=42)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.params = params or CalibrationParams()
        
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
    
    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Make a chat completion request with calibration parameters.
        
        Args:
            model: Model identifier
            messages: Chat messages [{role, content}, ...]
            tools: Optional tool definitions for function calling
            response_format: Optional response format (JSON mode)
            **kwargs: Additional parameters (override calibration defaults)
        
        Returns:
            Full API response as dict
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
            "max_tokens": self.params.max_tokens,
            **kwargs,
        }
        
        # Add seed if supported (not all providers support it)
        if self.params.seed is not None:
            payload["seed"] = self.params.seed
        
        if tools:
            payload["tools"] = tools
        
        if response_format:
            payload["response_format"] = response_format
        
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
    
    def complete_text(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        **kwargs,
    ) -> str:
        """
        Convenience method: get completion text from a prompt.
        
        Args:
            model: Model identifier
            prompt: User prompt
            system: Optional system message
            **kwargs: Additional parameters
        
        Returns:
            Assistant response text
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.complete(model, messages, **kwargs)
        return response["choices"][0]["message"]["content"]
    
    def complete_with_tools(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict],
        tool_choice: str | dict = "auto",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Make a completion request with tool calling.
        
        Args:
            model: Model identifier
            messages: Chat messages
            tools: Tool definitions
            tool_choice: "auto", "none", "required", or specific tool
            **kwargs: Additional parameters
        
        Returns:
            Full API response (check message.tool_calls)
        """
        return self.complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
    
    def list_models(self) -> list[str]:
        """List available models from the API."""
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
