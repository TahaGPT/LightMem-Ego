from __future__ import annotations

import os
from typing import Any


LOCAL_QWEN_PROFILE = "local-qwen35"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def local_llm_enabled() -> bool:
    value = os.getenv("EM2MEM_LOCAL_LLM_ENABLED", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def local_chat_request_kwargs() -> dict[str, Any]:
    """Return OpenAI SDK kwargs needed by the local Qwen chat template."""
    if not local_llm_enabled():
        return {}
    non_thinking = os.getenv("EM2MEM_LOCAL_LLM_NON_THINKING", "1")
    if non_thinking.strip().lower() not in {"1", "true", "yes", "on"}:
        return {}
    return {
        "max_tokens": _env_int("EM2MEM_LOCAL_LLM_MAX_TOKENS", 4096),
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        }
    }


def merge_local_chat_request_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(kwargs)
    local_kwargs = local_chat_request_kwargs()
    if not local_kwargs:
        return merged

    if "max_tokens" not in merged and "max_completion_tokens" not in merged:
        merged["max_tokens"] = local_kwargs["max_tokens"]
    local_extra = dict(local_kwargs.get("extra_body") or {})
    existing_extra = dict(merged.get("extra_body") or {})
    local_template = dict(local_extra.get("chat_template_kwargs") or {})
    existing_template = dict(existing_extra.get("chat_template_kwargs") or {})
    existing_template.update(local_template)
    existing_extra.update(local_extra)
    existing_extra["chat_template_kwargs"] = existing_template
    merged["extra_body"] = existing_extra
    return merged


def local_cache_namespace() -> str:
    if not local_llm_enabled():
        return ""
    value = os.getenv("EM2MEM_LOCAL_LLM_CACHE_NAMESPACE", "local_qwen35")
    return value.strip().replace("-", "_")
