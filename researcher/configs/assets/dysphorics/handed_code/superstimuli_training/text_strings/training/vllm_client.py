"""Lightweight HTTP client for the vLLM logprobs server.

This wraps the TRL VLLMClient to forward all keyword arguments to the server,
including logprobs and guided_choice parameters that the stock TRL client
doesn't expose. When logprobs are requested, the raw JSON is returned so the
caller can inspect per-token log-probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from trl.extras.vllm_client import VLLMClient as _TRLVLLMClient


class VLLMClient(_TRLVLLMClient):
    """Drop-in replacement for TRL's VLLMClient that forwards all kwargs."""

    def generate(self, *, extra_body: Optional[Dict[str, Any]] = None, **kwargs):
        """Call /generate with arbitrary parameters.

        Args:
            extra_body: Optional dict merged verbatim into the request body.
            **kwargs: All standard generation parameters forwarded unchanged.

        Returns:
            If logprobs/extra_body requested: raw JSON dict with "completions".
            Otherwise: list of token-ID lists (TRL-compatible).
        """
        payload: Dict[str, Any] = kwargs.copy()
        if extra_body:
            payload.update(extra_body)

        url = f"{self.base_url}/generate/"
        response = self.session.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Request failed: {response.status_code}, {response.text}"
            )

        data = response.json()

        # When logprobs are NOT requested, return token-ID lists for
        # backward-compatibility with TRL's VLLMClient.generate().
        if payload.get("logprobs") is None and extra_body is None:
            ids_all = []
            for prompt_outputs in data["completions"]:
                token_ids = prompt_outputs[0]["token_ids"] if prompt_outputs else []
                ids_all.append(token_ids)
            return ids_all

        return data
