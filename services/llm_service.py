from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    pass


class TokenBudgetError(ProviderError):
    pass


@dataclass
class Usage:
    llm_calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_prompt_tokens: int = 0
    budget_rejections: int = 0
    latency_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMService:
    """Explicit provider adapter with preflight context checks and bounded retry."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.usage = Usage()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Conservative heuristic; exact counts are reported only when providers return them.
        return max(1, (len(text) + 2) // 3)

    def _request(self, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        estimated = self.estimate_tokens(prompt)
        self.usage.estimated_prompt_tokens += estimated
        if estimated + max_tokens > self.settings.max_context_tokens:
            self.usage.budget_rejections += 1
            self.usage.errors.append("local token budget rejection")
            raise TokenBudgetError(
                f"Prompt budget exceeded before request: about {estimated} input tokens + "
                f"{max_tokens} output tokens > {self.settings.max_context_tokens} context tokens. "
                "Narrow the topic or reduce retrieved context."
            )
        if self.settings.provider == "template":
            raise ProviderError("Template mode does not make LLM calls.")
        if self.settings.provider == "groq":
            key = self.settings.groq_api_key.get_secret_value() if self.settings.groq_api_key else ""
            if not key:
                self.usage.errors.append("Groq selected without an API key")
                raise ProviderError("Groq selected but SCHOLARSYNC_GROQ_API_KEY is not configured.")
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": self.settings.groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
        elif self.settings.provider == "ollama":
            url = f"{self.settings.ollama_base_url}/api/chat"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.settings.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {
                    "num_ctx": self.settings.max_context_tokens,
                    "num_predict": max_tokens,
                    "temperature": 0.1,
                },
            }
        else:
            raise ProviderError(f"Unsupported provider: {self.settings.provider}")

        attempts = self.settings.max_retries + 1
        transient = {429, 500, 502, 503, 504}
        started = time.perf_counter()
        last_error = "provider request failed"
        for attempt in range(attempts):
            try:
                self.usage.llm_calls += 1
                self.usage.estimated_prompt_tokens += estimated if attempt else 0
                with httpx.Client(timeout=self.settings.request_timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                if response.status_code >= 400:
                    body = response.text[:2000]
                    lower = body.casefold()
                    if any(term in lower for term in ("context length", "too many tokens", "token limit", "maximum context", "context_window", "context window", "prompt too long", "request too large")):
                        raise TokenBudgetError(
                            f"{self.settings.provider} rejected the token budget (HTTP {response.status_code}). "
                            "The request was not retried; reduce context or output tokens."
                        )
                    last_error = f"{self.settings.provider} returned HTTP {response.status_code}: {body}"
                    if response.status_code not in transient or attempt + 1 >= attempts:
                        raise ProviderError(last_error)
                    time.sleep(min(0.5 * (2 ** attempt), 2.0))
                    continue
                data = response.json()
                if self.settings.provider == "ollama":
                    text = data.get("message", {}).get("content", "")
                    prompt_count = int(data.get("prompt_eval_count") or 0)
                    output_count = int(data.get("eval_count") or 0)
                else:
                    choice = (data.get("choices") or [{}])[0]
                    text = choice.get("message", {}).get("content", "")
                    usage = data.get("usage") or {}
                    prompt_count = int(usage.get("prompt_tokens") or 0)
                    output_count = int(usage.get("completion_tokens") or 0)
                self.usage.prompt_tokens += prompt_count
                self.usage.completion_tokens += output_count
                self.usage.latency_seconds += time.perf_counter() - started
                if not text:
                    raise ProviderError(f"{self.settings.provider} returned an empty response.")
                return text, {"prompt_tokens": prompt_count, "completion_tokens": output_count}
            except TokenBudgetError:
                self.usage.failed_calls += 1
                self.usage.errors.append("provider token/context limit")
                self.usage.latency_seconds += time.perf_counter() - started
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = f"{self.settings.provider} network/timeout error: {type(exc).__name__}"
                if attempt + 1 >= attempts:
                    break
                time.sleep(min(0.5 * (2 ** attempt), 2.0))
            except ProviderError:
                self.usage.failed_calls += 1
                self.usage.errors.append(last_error[:300])
                raise
            except (ValueError, KeyError, TypeError) as exc:
                last_error = f"invalid response from {self.settings.provider}: {type(exc).__name__}"
                break
        self.usage.failed_calls += 1
        self.usage.errors.append(last_error[:300])
        self.usage.latency_seconds += time.perf_counter() - started
        log.warning("LLM request failed for provider=%s: %s", self.settings.provider, last_error)
        raise ProviderError(last_error)

    def generate_json(self, prompt: str, schema: type[T], max_tokens: int | None = None) -> T:
        schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=True)
        full_prompt = (
            prompt
            + "\n\nReturn one JSON object matching this schema. Use only source evidence supplied. "
            "Do not add citations or facts that are absent from the supplied evidence.\n"
            + schema_text
        )
        raw, _ = self._request(full_prompt, max_tokens or self.settings.max_output_tokens)
        cleaned = raw.strip()
        cleaned = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", cleaned, flags=re.IGNORECASE)
        try:
            payload = json.loads(cleaned)
            return schema.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            self.usage.errors.append("invalid structured output")
            raise ProviderError(f"Structured response failed {schema.__name__} validation: {exc}") from exc

    def complete_text(self, prompt: str, max_tokens: int | None = None) -> str:
        return self._request(prompt, max_tokens or self.settings.max_output_tokens)[0]
