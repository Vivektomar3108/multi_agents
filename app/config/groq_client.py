# app/config/groq_client.py
import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Generator

logger = logging.getLogger(__name__)

# Prefer env variables in production
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_BASE_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
# Path: default /chat/completions (Groq uses an OpenAI-compatible path)
GROQ_CHAT_PATH = os.getenv("GROQ_CHAT_PATH", "/chat/completions")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment")

class GroqClient:
    """
    Minimal Groq/OpenAI-compatible client.
    - chat(...) returns parsed JSON dict for non-streaming calls
    - streaming returns generator of parsed JSON events (if stream=True)
    """

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        base = (api_url or GROQ_BASE_URL).rstrip("/")
        self.api_url = base
        self._chat_url = self.api_url + GROQ_CHAT_PATH
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        logger.debug("[GroqClient] initialized using url=%s", self._chat_url)

    def _send_chat_request(self, payload: Dict[str, Any], stream: bool = False) -> requests.Response:
        logger.debug("[GroqClient] POST %s payload_keys=%s stream=%s", self._chat_url, list(payload.keys()), stream)
        resp = self.session.post(self._chat_url, json=payload, stream=stream, timeout=180)
        try:
            resp.raise_for_status()
        except Exception:
            logger.error("[GroqClient] HTTP error: %s", resp.text)
            raise
        return resp

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        temperature: float = 0.0,
    ) -> Any:
        """
        Send chat request.

        Non-streaming: returns parsed JSON (dict).
        Streaming: returns a generator yielding parsed JSON event dicts.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            # Tools/functions schema — keep key name consistent with Groq/OpenAI function-calling schema
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if stream:
            return self._streaming_chat(payload)

        resp = self._send_chat_request(payload, stream=False)
        return resp.json()

    def _streaming_chat(self, payload) -> Generator[Dict[str, Any], None, None]:
        resp = self._send_chat_request(payload, stream=True)
        for raw in resp.iter_lines():
            if not raw:
                continue
            try:
                s = raw.decode("utf-8")
            except Exception:
                continue
            if s.startswith("data: "):
                chunk = s[len("data: "):]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(chunk)
                except Exception:
                    logger.debug("[GroqClient] malformed stream chunk: %s", chunk[:200])
                    continue
