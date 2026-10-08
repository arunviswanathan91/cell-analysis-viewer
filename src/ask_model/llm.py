"""Provider-agnostic chat client (OpenAI-compatible /chat/completions with tool calling).

Works with Groq, the Hugging Face Inference router and OpenAI. The model name is resolved
at runtime from the provider's own model list, so a retired model id (the failure that
broke the previous assistant) falls back to the next preferred model instead of erroring.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class LLMError(RuntimeError):
    """Raised with a message that is safe and useful to show the user."""


@dataclass(frozen=True)
class Provider:
    key: str            # short id
    label: str
    base_url: str
    secret_names: tuple  # secret / env var names that may hold the API key
    preferred: tuple     # model ids in order of preference (first one the provider serves wins)
    signup: str = ""


PROVIDERS: Dict[str, Provider] = {
    "groq": Provider(
        "groq", "Groq", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",),
        ("openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3-32b",
         "meta-llama/llama-4-scout-17b-16e-instruct", "openai/gpt-oss-20b", "llama-3.1-8b-instant"),
        "https://console.groq.com/keys"),
    "hf": Provider(
        "hf", "Hugging Face Inference", "https://router.huggingface.co/v1", ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
        ("openai/gpt-oss-120b", "Qwen/Qwen3-235B-A22B-Instruct-2507", "meta-llama/Llama-3.3-70B-Instruct",
         "Qwen/Qwen2.5-72B-Instruct", "openai/gpt-oss-20b", "Qwen/Qwen3-32B"),
        "https://huggingface.co/settings/tokens"),
    "openai": Provider(
        "openai", "OpenAI", "https://api.openai.com/v1", ("OPENAI_API_KEY",),
        ("gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4o-mini"), "https://platform.openai.com/api-keys"),
    # any OpenAI-compatible server (Ollama, vLLM, LM Studio, a proxy ...)
    "custom": Provider(
        "custom", "Custom endpoint", os.environ.get("ASK_MODEL_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
        ("ASK_MODEL_API_KEY",), (), ""),
}

_THINK = re.compile(r"<think>.*?</think>", re.S)


def _headers(key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _explain(resp: requests.Response, provider: Provider) -> str:
    try:
        body = resp.json()
        msg = body.get("error", body)
        msg = msg.get("message", msg) if isinstance(msg, dict) else msg
    except Exception:  # noqa: BLE001
        msg = resp.text[:300]
    hint = ""
    if resp.status_code in (401, 403):
        hint = f" Check that the {provider.label} API key is valid and has access."
    elif resp.status_code == 404:
        hint = " The selected model is not available to this key; choose another model."
    elif resp.status_code == 429:
        hint = " Rate limit reached; wait a moment or pick a smaller model."
    return f"{provider.label} returned HTTP {resp.status_code}: {str(msg)[:300]}.{hint}"


def list_models(provider: Provider, api_key: str, timeout: int = 20) -> List[str]:
    try:
        r = requests.get(f"{provider.base_url}/models", headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as e:
        raise LLMError(f"Could not reach {provider.label}: {e}") from e
    if r.status_code != 200:
        raise LLMError(_explain(r, provider))
    data = r.json().get("data", [])
    return sorted({m["id"] for m in data if isinstance(m, dict) and "id" in m})


def resolve_model(provider: Provider, available: List[str], wanted: Optional[str] = None) -> str:
    """Pick the requested model, else the first preferred model the provider serves."""
    if wanted and (wanted in available or not available):
        return wanted
    for m in provider.preferred:
        if m in available:
            return m
    chat_like = [m for m in available if not re.search(r"whisper|tts|guard|embed|prompt-guard|orpheus|distil-whisper", m, re.I)]
    if chat_like:
        return chat_like[0]
    raise LLMError(f"{provider.label} did not list any usable chat model for this key.")


def chat(provider: Provider, api_key: str, model: str, messages: List[Dict[str, Any]],
         tools: Optional[list] = None, temperature: float = 0.0, max_tokens: int = 1800,
         timeout: int = 90) -> Dict[str, Any]:
    """One completion. Returns the assistant message dict (content, tool_calls)."""
    payload: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        payload.update(tools=tools, tool_choice="auto")
    last = ""
    for attempt in range(3):
        try:
            r = requests.post(f"{provider.base_url}/chat/completions", headers=_headers(api_key),
                              data=json.dumps(payload), timeout=timeout)
        except requests.RequestException as e:
            last = f"Could not reach {provider.label}: {e}"
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            msg = r.json()["choices"][0]["message"]
            if isinstance(msg.get("content"), str):
                msg["content"] = _THINK.sub("", msg["content"]).strip()
            return msg
        last = _explain(r, provider)
        # transient, or the model produced a malformed tool call: try again
        if r.status_code in (429, 500, 502, 503, 504) or "tool_use_failed" in r.text or "Failed to call a function" in r.text:
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    raise LLMError(last or "Unknown LLM failure.")
