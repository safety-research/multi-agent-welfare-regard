from __future__ import annotations

"""Lightweight wrapper around ``trl.extras.vllm_client.VLLMClient`` that
exposes the **full** flexibility of the vLLM ``/generate`` endpoint.

The stock ``VLLMClient`` included in TRL only allows the subset of generation
parameters needed by the CLI.  For research experiments we often want to pass
additional parameters such as ``logprobs``, ``allowed_token_ids`` or the
experimental ``guided_choice`` option implemented in the custom vLLM server
(`vllm_serve_logprobs.py`).

This subclass forwards *all* keyword arguments it receives directly to the
server so you are never limited by the client-side API surface.

Example
-------
>>> from flexible_vllm_client import FlexibleVLLMClient
>>> client = FlexibleVLLMClient(host="0.0.0.0", server_port=8000)
>>> response = client.generate(
...     prompts=["Hello"],
...     n=1,
...     max_tokens=1,
...     logprobs=2,
...     allowed_token_ids=[123, 456],
... )
>>> print(response)
{"completions": [[{...}]]}
"""

from typing import Any, Dict, Optional

from trl.extras.vllm_client import VLLMClient as _TRLVLLMClient

__all__ = ["FlexibleVLLMClient"]


class VLLMClient(_TRLVLLMClient):
    """A drop-in replacement for :class:`~trl.extras.vllm_client.VLLMClient` that
    forwards *all* keyword arguments to the server's ``/generate`` endpoint.

    It also supports an optional ``extra_body`` argument which is merged into
    the JSON payload verbatim – handy for parameters that are not part of the
    official TRL client interface (e.g. ``guided_choice``).
    """

    def generate(self, *, extra_body: Optional[Dict[str, Any]] = None, **kwargs):  # noqa: D401
        """Call the server's ``/generate`` endpoint with arbitrary parameters.

        Parameters
        ----------
        extra_body:
            Optional dictionary merged *verbatim* into the request body.
        **kwargs:
            All standard generation parameters are forwarded unchanged.
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

        # Backward-compatibility: when *logprobs* is **not** requested we mimic
        # the behaviour of the original TRL ``VLLMClient.generate`` and return
        # a list of token-ID lists.  Otherwise we return the full JSON dict so
        # the caller can inspect the ``logprobs`` field.
        if payload.get("logprobs") is None and extra_body is None:
            # Extract the token_ids from the first completion of each prompt.
            ids_all = []
            for prompt_outputs in data["completions"]:
                # Each prompt_outputs is a list (n completions). Take first.
                token_ids = prompt_outputs[0]["token_ids"] if prompt_outputs else []
                ids_all.append(token_ids)
            return ids_all

        # If logprobs or other advanced feature requested, return raw JSON
        return data 