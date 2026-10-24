"""Prove which model serves our requests: raw HTTP, no SDK parsing.

Makes one tiny chat completion call per model name and prints the `model`
field the SERVER echoes back in the response body, plus the response id.
If the harness were secretly using another model, the echoed field would
not read "mistral-large-4".

Run: .venv/Scripts/python tools/verify_model.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

API = "https://api.mistral.ai/v1/chat/completions"


def call(model: str):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 4,
    }).encode()
    req = urllib.request.Request(
        API,
        data=payload,
        headers={
            "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return f"<HTTP {e.code}: {e.read()[:120].decode()}>", None, None
    usage = body.get("usage") or {}
    return (
        body.get("model"),
        body.get("id"),
        f"prompt {usage.get('prompt_tokens')} / completion {usage.get('completion_tokens')}",
    )


print("Raw HTTP POST to api.mistral.ai/v1/chat/completions\n")
for model in ("mistral-large-4", "mistral-large-latest"):
    served, rid, usage = call(model)
    print(f"requested model: {model}")
    print(f"  -> served by  : {served}")
    print(f"  -> response id: {rid}")
    print(f"  -> usage      : {usage}\n")
