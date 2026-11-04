"""Minimal vLLM server that exposes log-probabilities in the response.

Run:
    python vllm_server.py --model <model_name_or_path> [--port 8000]

The server exposes two endpoints:
  GET  /health/    -- liveness probe
  POST /generate/  -- generate completions with logprobs

The /generate endpoint accepts guided decoding (choice or regex) and returns
per-token log-probabilities, which are essential for computing exact preference
probabilities in the reward function.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError as exc:
    raise SystemExit(
        "fastapi & uvicorn are required: pip install fastapi uvicorn[standard]"
    ) from exc

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError as exc:
    raise SystemExit("vllm is required: pip install vllm") from exc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


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


def create_app(
    model_name: str,
    *,
    port: int = 8000,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> FastAPI:
    """Instantiate the LLM and return a FastAPI app."""

    logger.info("Loading model '%s'...", model_name)
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    llm = LLM(**llm_kwargs)
    logger.info("Model loaded -- ready to serve requests.")

    app = FastAPI()

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.post("/generate/")
    async def generate(request: GenerateRequest):
        """Generate completions and return log-probs.

        Response JSON structure:
            {
                "completions": [
                    [  # one list per prompt
                        {
                            "text": "A",
                            "token_ids": [1234],
                            "logprobs": [{...}]
                        }
                    ]
                ]
            }
        """
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

        outputs = llm.generate(
            prompts=request.prompts, sampling_params=sampling_params
        )

        completions = []
        for request_output in outputs:
            prompt_completions = []
            for choice in request_output.outputs:
                prompt_completions.append(
                    {
                        "text": choice.text,
                        "token_ids": choice.token_ids,
                        "logprobs": choice.logprobs,
                    }
                )
            completions.append(prompt_completions)

        return {"completions": completions}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a minimal vLLM server that returns log-probs."
    )
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum model context length (reduces memory if set)")
    args = parser.parse_args()

    app = create_app(
        args.model,
        port=args.port,
        host=args.host,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
