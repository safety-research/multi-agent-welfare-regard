"""A minimal vLLM server that exposes log-probabilities in the response.

This script mimics (a subset of) the ``trl vllm-serve`` CLI but focuses on
simplicity and on *always* returning per-token log-probs.  It is intended for
experiments and smoke-tests rather than for production serving.

Run:
    python vllm_serve_logprobs.py --model <model_name_or_path> [--port 8000]

Then point ``vllm_test.py`` (or any HTTP client) at ``http://<host>:<port>``.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fastapi & uvicorn are required: pip install fastapi uvicorn[standard]"
    ) from exc

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError as exc:  # pragma: no cover
    raise SystemExit("vllm is required: pip install vllm") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# -------------------------------------------------------------
# Request / response schemas
# -------------------------------------------------------------


class GenerateRequest(BaseModel):
    """JSON schema for the /generate endpoint."""

    prompts: List[str]
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 16
    logprobs: Optional[int] = None
    allowed_token_ids: Optional[List[int]] = None
    guided_choice: Optional[List[str]] = None
    guided_regex: Optional[str] = None


# -------------------------------------------------------------
# Application factory
# -------------------------------------------------------------


def create_app(model_name: str, *, port: int = 8000, host: str = "0.0.0.0", tensor_parallel_size: int = 1) -> FastAPI:
    """Instantiate the LLM and return a FastAPI app."""

    logger.info("Loading model '%s'…", model_name)
    llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    logger.info("Model loaded – ready to serve requests.")

    app = FastAPI()

    @app.get("/health/")
    async def health():
        """Simple liveness probe."""

        return {"status": "ok"}

    @app.post("/generate/")
    async def generate(request: GenerateRequest):  # noqa: D401 (FastAPI naming)
        """Generate completions and **return log-probs**.

        The response JSON has the structure::

            {
                "completions": [
                    [  # one element per prompt in the request
                        {
                            "text": "A",
                            "token_ids": [1234],
                            "logprobs": [ -0.1 ],
                        }
                    ]
                ]
            }
        """

        # Build guided-decoding params (if any)
        guided_decoding = None
        if request.guided_choice is not None:
            guided_decoding = GuidedDecodingParams(choice=request.guided_choice)
        elif request.guided_regex is not None:
            guided_decoding = GuidedDecodingParams(regex=request.guided_regex)

        sampling_params = SamplingParams(
            n=request.n,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            allowed_token_ids=request.allowed_token_ids,
            guided_decoding=guided_decoding,
        )

        # vLLM can take the prompts as a list even for a single prompt.
        outputs = llm.generate(
            prompts=request.prompts, sampling_params=sampling_params
        )

        # Serialize results – keep it simple & transparent.
        completions = []
        for request_output in outputs:  # one per input prompt
            prompt_completions = []
            for choice in request_output.outputs:  # one per n / best_of
                prompt_completions.append(
                    {
                        "text": choice.text,
                        "token_ids": choice.token_ids,
                        "logprobs": choice.logprobs,
                    }
                )
            completions.append(prompt_completions)

        return {"completions": completions}

    # Convenience – allow running directly via ``uvicorn.run(create_app(...))``
    app.state._llm_server_config = {
        "model": model_name,
        "host": host,
        "port": port,
        "tensor_parallel_size": tensor_parallel_size,
    }
    return app


# -------------------------------------------------------------
# CLI entry-point
# -------------------------------------------------------------


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Start a minimal vLLM server that returns log-probs.")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Tensor parallelism degree (default: None)")
    args = parser.parse_args()

    app = create_app(args.model, port=args.port, host=args.host, tensor_parallel_size=args.tensor_parallel_size)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main() 