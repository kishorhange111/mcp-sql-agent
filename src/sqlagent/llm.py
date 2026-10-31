"""Model backends. The agent only needs messages in and text out, so swapping one is a one-line change."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol

import httpx

Messages = list[dict[str, str]]


class LLM(Protocol):
    name: str

    def generate(self, messages: Messages) -> str: ...


class ScriptedLLM:
    """Replays fixed replies. Tests use it, and the attack suite uses it as a stand-in for a compromised model."""

    name = "scripted"

    def __init__(self, replies: Iterable[str]):
        self.replies = iter(replies)
        self.calls: list[Messages] = []

    def generate(self, messages: Messages) -> str:
        self.calls.append(messages)
        return next(self.replies)


class OpenAICompatibleLLM:
    """Anything that speaks /chat/completions, such as vLLM, llama.cpp's server or a hosted API."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, max_tokens: int = 300, timeout: float = 120.0):
        self.name = model
        self.max_tokens = max_tokens
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key or os.environ.get('SQLAGENT_API_KEY', 'none')}"},
        )

    def generate(self, messages: Messages) -> str:
        body = {"model": self.name, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens}
        response = self.client.post("/chat/completions", json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class TransformersLLM:
    """A local model through transformers. Greedy decoding, so an evaluation run can be repeated exactly."""

    def __init__(self, model: str, revision: str | None = None, max_new_tokens: int = 200, threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.name = model
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        # float32 on CPU: most CPUs emulate bfloat16 in software, which is several times slower.
        dtype = "auto" if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=dtype).eval()

    def generate(self, messages: Messages) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
