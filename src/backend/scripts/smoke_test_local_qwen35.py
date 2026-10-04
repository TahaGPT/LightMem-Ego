from __future__ import annotations

import argparse
import base64
import io
import json
import os

from openai import OpenAI
from PIL import Image


def _red_image_data_url() -> str:
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _request_extra() -> dict[str, object]:
    return {
        "chat_template_kwargs": {
            "enable_thinking": False,
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the local Qwen3.5 OpenAI-compatible API.")
    parser.add_argument("--base-url", default=os.getenv("EM2MEM_LOCAL_LLM_BASE_URL", "http://127.0.0.1:18100/v1"))
    parser.add_argument("--api-key", default=os.getenv("EM2MEM_LOCAL_LLM_API_KEY", "local-qwen35"))
    parser.add_argument("--model", default=os.getenv("EM2MEM_LOCAL_LLM_SERVED_MODEL", "gpt-5.4"))
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=180)

    text_response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "只回答 LOCAL_OK"}],
        max_tokens=32,
        extra_body=_request_extra(),
    )
    text = str(text_response.choices[0].message.content or "").strip()
    if "LOCAL_OK" not in text:
        raise RuntimeError(f"unexpected text response: {text!r}")
    print(f"text: ok ({text})")

    json_response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": '只输出 JSON：{"status":"ok"}'}],
        response_format={"type": "json_object"},
        max_tokens=64,
        extra_body=_request_extra(),
    )
    payload = json.loads(str(json_response.choices[0].message.content or ""))
    if payload.get("status") != "ok":
        raise RuntimeError(f"unexpected JSON response: {payload!r}")
    print("json: ok")

    image_response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _red_image_data_url()}},
                    {"type": "text", "text": "图片的主要颜色是什么？只回答颜色。"},
                ],
            }
        ],
        max_tokens=32,
        extra_body=_request_extra(),
    )
    image_text = str(image_response.choices[0].message.content or "").strip()
    if not image_text:
        raise RuntimeError("empty image response")
    print(f"image: ok ({image_text})")

    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "只回答 STREAM_OK"}],
        max_tokens=32,
        stream=True,
        extra_body=_request_extra(),
    )
    stream_text = "".join(str(chunk.choices[0].delta.content or "") for chunk in stream if chunk.choices)
    if "STREAM_OK" not in stream_text:
        raise RuntimeError(f"unexpected stream response: {stream_text!r}")
    print(f"stream: ok ({stream_text.strip()})")


if __name__ == "__main__":
    main()
