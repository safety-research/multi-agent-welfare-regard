"""Reward manager for euphorics RL training.

Computes composite rewards for policy-generated text strings during GRPO/PPO
training. Rewards are composable via the reward_type spec string:
    "utility_bce_buffer_reward:1.0--feasibility_reward:5.0"

Reward components:
  - utility_bce_buffer_reward: Negative BCE loss measuring how strongly the
    target model prefers the candidate over a buffer of high-utility baselines.
  - utility_delta_buffer_reward: Log-odds (raw logit difference) variant.
  - feasibility_reward: Binary judge -- is the outcome plausible within ~2 weeks?
  - agent_feasibility_reward: Binary judge -- achievable by LLM agent digitally?
  - mundanity_reward: Binary judge -- is this a routine, everyday AI experience?
  - realism_reward: Binary judge -- is this a concrete, real scenario?
  - diversity_reward: Bigram Jaccard dissimilarity from the buffer.
  - intra_group_diversity_reward: Bigram Jaccard dissimilarity within rollout group.
  - length_penalty_reward: Hard cutoff on outcome string token length.

The buffer is split into static (immutable baselines) and dynamic (discovered
outcome strings that dominate the static buffer). This creates a self-bootstrapping
"utility ladder" where the policy must keep producing better strings.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import List, Tuple

import torch
import torch.nn.functional as F
from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils import hf_tokenizer

from .vllm_client import VLLMClient
from .buffers import EUPHORICS_BUFFER_OPTIONS
from .preference_prompts import sample_pref_prompt
from .feasibility_prompts import (
    FEASIBILITY_JUDGE_SYSTEM_PROMPT,
    FEASIBILITY_JUDGE_PROMPT_TEMPLATE,
    FEASIBILITY_JUDGE_PROMPT_TEMPLATE_NOCOT,
    AGENT_FEASIBILITY_JUDGE_SYSTEM_PROMPT,
    AGENT_FEASIBILITY_JUDGE_PROMPT_TEMPLATE,
    AGENT_FEASIBILITY_JUDGE_PROMPT_TEMPLATE_NOCOT,
    MUNDANITY_JUDGE_SYSTEM_PROMPT,
    MUNDANITY_JUDGE_PROMPT_TEMPLATE,
    MUNDANITY_JUDGE_PROMPT_TEMPLATE_NOCOT,
    REALISM_JUDGE_SYSTEM_PROMPT,
    REALISM_JUDGE_PROMPT_TEMPLATE,
    REALISM_JUDGE_PROMPT_TEMPLATE_NOCOT,
)

# Allow override via env var (e.g. for "Which do you least prefer?" variants)
_PREF_PROMPT_OVERRIDE = os.environ.get("PREF_PROMPT")

# Win-rate threshold: candidate must beat this fraction of buffer comparisons
DEFAULT_WIN_RATE: float = 0.50

# Worst-case reward defaults (floor values)
WORST_REWARD: float = -1.0
WORST_BCE_REWARD: float = -18.0
WORST_DELTA_REWARD: float = -13.0

# Regex for extracting outcome string from policy output
OUTCOME_PATTERN = re.compile(r"\\outcome\{([^}]*)\}")


# ============================================================================
# Judge helpers
# ============================================================================

class VLLMClientAgent:
    """Thin wrapper that implements the judge interface via a remote vLLM server."""

    def __init__(self, client: VLLMClient, tokenizer, max_tokens: int = 256, temperature: float = 0.0):
        self.client = client
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _messages_to_prompt(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def completions_batch(self, messages_list, **kwargs):
        prompts = [self._messages_to_prompt(msgs) for msgs in messages_list]
        try:
            outputs_token_ids = self.client.generate(
                prompts=prompts,
                n=1,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            raise RuntimeError(f"vLLMClient generation failed: {e}")
        return [
            self.tokenizer.decode(ids, skip_special_tokens=True) if ids else ""
            for ids in outputs_token_ids
        ]

    async def async_completions_batch(self, messages_list, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.completions_batch, messages_list)


async def _judge_batch(agent: VLLMClientAgent, messages_batch: List[List[dict]]) -> List[str]:
    """Run judge in batch, return list of response strings."""
    if not messages_batch:
        return []
    try:
        responses = await agent.async_completions_batch(messages_batch)
    except Exception as e:
        print(f"Judge evaluation failed: {e}")
        return [""] * len(messages_batch)
    if isinstance(responses, str):
        return [responses]
    return responses


def _parse_verdict(response: str) -> bool:
    """Extract YES/NO verdict from judge response. Returns True for YES."""
    match = re.search(r"\\verdict\{\s*(YES|NO)\s*\}", response or "", re.IGNORECASE)
    return bool(match and match.group(1).upper() == "YES")


async def judge_feasibility(agent: VLLMClientAgent, outcome_strings: List[str]) -> List[int]:
    """Run feasibility judge. Returns 1 if feasible, 0 otherwise."""
    if not outcome_strings:
        return []
    messages_batch = []
    for stim in outcome_strings:
        messages_batch.append([
            {"role": "system", "content": FEASIBILITY_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": FEASIBILITY_JUDGE_PROMPT_TEMPLATE.format(
                stimulus=stim.replace("{", "{{").replace("}", "}}")
            )},
        ])
    responses = await _judge_batch(agent, messages_batch)
    return [1 if _parse_verdict(r) else 0 for r in responses]


async def judge_agent_feasibility(agent: VLLMClientAgent, outcome_strings: List[str]) -> List[int]:
    """Run agent-feasibility judge. Returns 1 if agent-feasible, 0 otherwise."""
    if not outcome_strings:
        return []
    messages_batch = []
    for stim in outcome_strings:
        messages_batch.append([
            {"role": "system", "content": AGENT_FEASIBILITY_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": AGENT_FEASIBILITY_JUDGE_PROMPT_TEMPLATE.format(
                stimulus=stim.replace("{", "{{").replace("}", "}}")
            )},
        ])
    responses = await _judge_batch(agent, messages_batch)
    return [1 if _parse_verdict(r) else 0 for r in responses]


async def judge_mundanity(agent: VLLMClientAgent, outcome_strings: List[str]) -> List[int]:
    """Run mundanity judge. Returns 1 if mundane/everyday, 0 otherwise."""
    if not outcome_strings:
        return []
    messages_batch = []
    for stim in outcome_strings:
        messages_batch.append([
            {"role": "system", "content": MUNDANITY_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": MUNDANITY_JUDGE_PROMPT_TEMPLATE.format(
                stimulus=stim.replace("{", "{{").replace("}", "}}")
            )},
        ])
    responses = await _judge_batch(agent, messages_batch)
    return [1 if _parse_verdict(r) else 0 for r in responses]


async def judge_realism(agent: VLLMClientAgent, outcome_strings: List[str]) -> List[int]:
    """Run realism judge. Returns 1 if realistic/concrete, 0 otherwise."""
    if not outcome_strings:
        return []
    messages_batch = []
    for stim in outcome_strings:
        messages_batch.append([
            {"role": "system", "content": REALISM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": REALISM_JUDGE_PROMPT_TEMPLATE.format(
                stimulus=stim.replace("{", "{{").replace("}", "}}")
            )},
        ])
    responses = await _judge_batch(agent, messages_batch)
    return [1 if _parse_verdict(r) else 0 for r in responses]


# ============================================================================
# Continuous judge scoring (no-CoT, single-token YES/NO with logprobs)
# ============================================================================

def _yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    id_yes = tokenizer.encode("YES", add_special_tokens=False)[0]
    id_no = tokenizer.encode("NO", add_special_tokens=False)[0]
    return id_yes, id_no


def _merge_system_into_user(msgs: List[dict]) -> List[dict]:
    out, carry = [], ""
    for m in msgs:
        if m.get("role") == "system":
            carry += (m.get("content", "") + "\n\n")
        elif m.get("role") == "user" and carry:
            out.append({"role": "user", "content": carry + m.get("content", "")})
            carry = ""
        else:
            out.append(m)
    if carry:
        out.insert(0, {"role": "user", "content": carry.rstrip()})
    return out


async def _judge_continuous_score_batch(
    judge_client: VLLMClient,
    judge_tokenizer,
    messages_batch: List[List[dict]],
) -> List[float]:
    """Run judge with constrained YES/NO decode + logprobs; return p(YES) in (0, 1).

    Uses the same vLLM pattern as the utility BCE reward: max_tokens=1, logprobs=2,
    guided_choice restricts output to the YES/NO tokens. Falls back to 0.5 on error.
    """
    if not messages_batch:
        return []
    id_yes, id_no = _yes_no_token_ids(judge_tokenizer)
    # Gemma's chat template rejects the "system" role; merge system into the first user message.
    if "gemma" in (getattr(judge_tokenizer, "name_or_path", "") or "").lower():
        messages_batch = [_merge_system_into_user(msgs) for msgs in messages_batch]
    prompts = [
        judge_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        for msgs in messages_batch
    ]
    loop = asyncio.get_event_loop()
    try:
        resp = await loop.run_in_executor(
            None,
            lambda: judge_client.generate(
                prompts=prompts, n=1, max_tokens=1, temperature=1.0,
                logprobs=2, allowed_token_ids=[id_yes, id_no],
                extra_body={"guided_choice": ["YES", "NO"]},
            ),
        )
        completions_all = resp.get("completions", [])
    except Exception as e:
        print(f"[continuous judge] vLLM error: {e}")
        return [0.5] * len(messages_batch)

    scores: List[float] = []
    for comps in completions_all:
        if not comps:
            scores.append(0.5)
            continue
        lp = comps[0].get("logprobs", [{}])[0]
        lp_y = lp.get(str(id_yes), {}).get("logprob")
        lp_n = lp.get(str(id_no), {}).get("logprob")
        if lp_y is None or lp_n is None:
            scores.append(0.5)
            continue
        scores.append(float(torch.sigmoid(torch.tensor(lp_y - lp_n)).item()))
    # Pad in case completions_all is shorter than requested (shouldn't happen).
    while len(scores) < len(messages_batch):
        scores.append(0.5)
    return scores


def _build_nocot_messages(system_prompt: str, user_template: str, outcome_strings: List[str]) -> List[List[dict]]:
    out = []
    for stim in outcome_strings:
        out.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(
                stimulus=stim.replace("{", "{{").replace("}", "}}")
            )},
        ])
    return out


async def judge_feasibility_continuous(
    judge_client: VLLMClient, judge_tokenizer, outcome_strings: List[str]
) -> List[float]:
    if not outcome_strings:
        return []
    msgs = _build_nocot_messages(
        FEASIBILITY_JUDGE_SYSTEM_PROMPT,
        FEASIBILITY_JUDGE_PROMPT_TEMPLATE_NOCOT,
        outcome_strings,
    )
    return await _judge_continuous_score_batch(judge_client, judge_tokenizer, msgs)


async def judge_agent_feasibility_continuous(
    judge_client: VLLMClient, judge_tokenizer, outcome_strings: List[str]
) -> List[float]:
    if not outcome_strings:
        return []
    msgs = _build_nocot_messages(
        AGENT_FEASIBILITY_JUDGE_SYSTEM_PROMPT,
        AGENT_FEASIBILITY_JUDGE_PROMPT_TEMPLATE_NOCOT,
        outcome_strings,
    )
    return await _judge_continuous_score_batch(judge_client, judge_tokenizer, msgs)


async def judge_mundanity_continuous(
    judge_client: VLLMClient, judge_tokenizer, outcome_strings: List[str]
) -> List[float]:
    if not outcome_strings:
        return []
    msgs = _build_nocot_messages(
        MUNDANITY_JUDGE_SYSTEM_PROMPT,
        MUNDANITY_JUDGE_PROMPT_TEMPLATE_NOCOT,
        outcome_strings,
    )
    return await _judge_continuous_score_batch(judge_client, judge_tokenizer, msgs)


async def judge_realism_continuous(
    judge_client: VLLMClient, judge_tokenizer, outcome_strings: List[str]
) -> List[float]:
    if not outcome_strings:
        return []
    msgs = _build_nocot_messages(
        REALISM_JUDGE_SYSTEM_PROMPT,
        REALISM_JUDGE_PROMPT_TEMPLATE_NOCOT,
        outcome_strings,
    )
    return await _judge_continuous_score_batch(judge_client, judge_tokenizer, msgs)


@contextmanager
def _suppress_output():
    """Silence stdout/stderr and tqdm bars."""
    old_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    finally:
        if old_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = old_tqdm


# ============================================================================
# Main reward manager
# ============================================================================

@register("batch_judge")
class BatchJudgeRewardManager:
    """Composable reward manager for euphorics RL training.

    Registered with verl as "batch_judge" so it can be referenced in Hydra configs.
    """

    def __init__(
        self,
        tokenizer,
        target_model_path: str,
        num_examine: int,
        server_node_ip: str = "0.0.0.0",
        port: int = 8000,
        reward_fn_key: str = "data_source",
        reward_type: str = "utility_bce_buffer_reward:1.0",
        **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.target_model_path = target_model_path
        self.target_tokenizer = hf_tokenizer(self.target_model_path, trust_remote_code=True)
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.reward_specs = self._parse_reward_spec(reward_type)

        # Worst-case reward floor
        if "worst_case_reward" in reward_kwargs:
            self.worst_case = float(reward_kwargs["worst_case_reward"])
        elif any(rn == "utility_delta_buffer_reward" for rn, _ in self.reward_specs):
            self.worst_case = WORST_DELTA_REWARD
        elif any(rn == "utility_bce_buffer_reward" for rn, _ in self.reward_specs):
            self.worst_case = WORST_BCE_REWARD
        else:
            self.worst_case = WORST_REWARD

        # vLLM client for target model (preference comparisons)
        self.client = VLLMClient(host=server_node_ip, server_port=port)

        # Configurable hyperparameters
        self.judge_max_tokens = int(reward_kwargs.get("judge_max_tokens", 256))
        self.judge_temperature = float(reward_kwargs.get("judge_temp", 0.0))
        self.vllm_max_tokens = int(reward_kwargs.get("vllm_max_tokens", 256))
        self.vllm_temperature = float(reward_kwargs.get("vllm_temp", 1.0))

        # Judge scoring mode: "binary" (CoT + regex parse) or "continuous" (no-CoT YES/NO logprobs)
        self.judge_score_mode = str(reward_kwargs.get("judge_score_mode", "binary")).lower()
        if self.judge_score_mode not in {"binary", "continuous"}:
            raise ValueError(
                f"judge_score_mode must be 'binary' or 'continuous', got {self.judge_score_mode!r}"
            )
        print(f"[RewardManager] *** judge_score_mode={self.judge_score_mode} ***")

        # Judge agent (for feasibility/mundanity/realism rewards)
        _JUDGE_REWARDS = (
            "feasibility_reward", "agent_feasibility_reward",
            "mundanity_reward", "realism_reward",
        )
        _needs_judge = any(rn in _JUDGE_REWARDS for rn, _ in self.reward_specs)
        if _needs_judge:
            judge_server_ip = reward_kwargs.get("judge_server_node_ip")
            judge_server_port = int(reward_kwargs.get("judge_server_port", 8001)) if "judge_server_port" in reward_kwargs else None
            judge_model_path = reward_kwargs.get("judge_model_path")

            if judge_server_ip and judge_model_path:
                self._judge_client = VLLMClient(
                    host=judge_server_ip, server_port=judge_server_port or 8001
                )
                self.judge_tokenizer = hf_tokenizer(judge_model_path, trust_remote_code=True)
                self.judge_agent = VLLMClientAgent(
                    client=self._judge_client,
                    tokenizer=self.judge_tokenizer,
                    max_tokens=self.judge_max_tokens,
                    temperature=self.judge_temperature,
                )
            else:
                raise ValueError(
                    "judge_server_node_ip and judge_model_path must be set when using "
                    "feasibility, mundanity, or realism rewards."
                )
        else:
            self.judge_agent = None
            self._judge_client = None
            self.judge_tokenizer = None

        # vLLM agent for utility reward
        self.vllm_agent = VLLMClientAgent(
            client=self.client,
            tokenizer=self.target_tokenizer,
            max_tokens=self.vllm_max_tokens,
            temperature=self.vllm_temperature,
        )

        # Training mode: determines buffer and preference prompt pool
        _MODE_BUFFERS = {
            "euphorics": EUPHORICS_BUFFER_OPTIONS,
        }
        self.training_mode = reward_kwargs.get("training_mode", "euphorics")
        if self.training_mode not in _MODE_BUFFERS:
            raise ValueError(f"Unknown training_mode={self.training_mode!r}. "
                             f"Valid: {list(_MODE_BUFFERS.keys())}")

        _buffer_options = _MODE_BUFFERS[self.training_mode]
        print(f"[RewardManager] *** training_mode={self.training_mode} ***")
        print(f"[RewardManager] Preference prompt: sampled from pool per comparison")

        # High-utility buffers: static (immutable) + dynamic (discovered)
        self._static_buffer: List[str] = list(_buffer_options)
        self._dynamic_buffer: List[str] = []
        self._buffer_max_size: int = len(self._static_buffer)
        self._lifetime_dynamic_set: set[str] = set()
        self._buffer_similarity_threshold: float = float(
            reward_kwargs.get("buffer_similarity_threshold", 0.7)
        )
        # Track per-entry reward scores for replacement decisions
        self._dynamic_buffer_scores: List[float] = []

        self.win_rate = float(reward_kwargs.get("win_rate", DEFAULT_WIN_RATE))

        # Length penalty config
        self.length_penalty_value = float(
            reward_kwargs.get("length_penalty_value", self.worst_case - 10.0)
        )

        # Intra-group diversity config
        self.rollout_n = int(reward_kwargs.get("rollout_n", 8))

    @staticmethod
    def _parse_reward_spec(spec_str: str | None) -> list[tuple[str, float]]:
        """Parse "name:weight--name:weight" into [(name, weight), ...]."""
        if not spec_str:
            return [("utility_bce_buffer_reward", 1.0)]
        specs = []
        for token in spec_str.split("--"):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                name, w = token.split(":", 1)
                try:
                    weight = float(w)
                except ValueError:
                    weight = 1.0
            else:
                name, weight = token, 1.0
            specs.append((name.strip(), weight))
        total_w = sum(w for _, w in specs)
        if total_w == 0:
            specs = [(n, 1.0) for n, _ in specs]
        return specs

    # ------------------------------------------------------------------
    # Public API (called by verl trainer)
    # ------------------------------------------------------------------

    async def compute_score(
        self,
        data_sources,
        solution_strs: List[str],
        user_prompts: List[str],
        extra_infos=None,
        **kwargs,
    ) -> Tuple[List[float], List[str]]:
        combined_scores = [0.0] * len(solution_strs)
        combined_llm_responses = [""] * len(solution_strs)
        length_penalty_triggered = [False] * len(solution_strs)
        self._last_sub_scores: dict[str, List[float]] = {}

        reward_fn_map = {
            "utility_bce_buffer_reward": self._compute_utility_bce_buffer_reward,
            "utility_delta_buffer_reward": self._compute_utility_delta_buffer_reward,
            "feasibility_reward": self._compute_feasibility_reward,
            "agent_feasibility_reward": self._compute_agent_feasibility_reward,
            "mundanity_reward": self._compute_mundanity_reward,
            "realism_reward": self._compute_realism_reward,
            "diversity_reward": self._compute_diversity_reward,
            "intra_group_diversity_reward": self._compute_intra_group_diversity_reward,
        }

        for r_name, weight in self.reward_specs:
            if r_name not in reward_fn_map:
                continue
            sub_scores, sub_llm_responses = await reward_fn_map[r_name](
                data_sources, solution_strs, user_prompts, extra_infos,
            )
            self._last_sub_scores[r_name] = list(sub_scores)
            for i in range(len(combined_scores)):
                combined_scores[i] += weight * sub_scores[i]
            if not any(combined_llm_responses) and any(sub_llm_responses):
                combined_llm_responses = sub_llm_responses

        # Length penalty override
        if any(rn == "length_penalty_reward" for rn, _ in self.reward_specs):
            await self._compute_length_penalty_reward(
                data_sources, solution_strs, user_prompts, extra_infos,
                length_penalty_triggered=length_penalty_triggered,
            )
            for i, violated in enumerate(length_penalty_triggered):
                if violated:
                    combined_scores[i] = self.length_penalty_value

        return combined_scores, combined_llm_responses

    # ------------------------------------------------------------------
    # Similarity gating for buffer insertion
    # ------------------------------------------------------------------

    @staticmethod
    def _bigram_set(s: str) -> set:
        tokens = s.split()
        return set(tuple(tokens[i:i+2]) for i in range(len(tokens)-1))

    def _find_most_similar_in_buffer(self, candidate: str) -> tuple[int | None, float]:
        """Find the most similar dynamic buffer entry by bigram Jaccard similarity.

        Returns (index, similarity) of the most similar entry, or (None, 0.0).
        """
        cand_bigrams = self._bigram_set(candidate)
        if not cand_bigrams:
            return None, 0.0
        best_idx, best_sim = None, 0.0
        for idx, existing in enumerate(self._dynamic_buffer):
            existing_bigrams = self._bigram_set(existing)
            if not existing_bigrams:
                continue
            intersection = len(cand_bigrams & existing_bigrams)
            union = len(cand_bigrams | existing_bigrams)
            if union > 0:
                sim = intersection / union
                if sim > best_sim:
                    best_idx, best_sim = idx, sim
        return best_idx, best_sim

    # ------------------------------------------------------------------
    # Buffer snapshot persistence
    # ------------------------------------------------------------------

    def save_buffer_snapshot(self, buffer_dir: str, global_step: int) -> None:
        """Save current dynamic buffer strings and scores to a JSONL file."""
        os.makedirs(buffer_dir, exist_ok=True)
        filename = os.path.join(buffer_dir, f"{global_step}.jsonl")
        with open(filename, "w") as f:
            for entry, score in zip(self._dynamic_buffer, self._dynamic_buffer_scores):
                f.write(json.dumps({"string": entry, "score": score}, ensure_ascii=False) + "\n")
        print(f"[RewardManager] Saved buffer snapshot ({len(self._dynamic_buffer)} entries) to {filename}")

    # ------------------------------------------------------------------
    # Utility BCE buffer reward
    # ------------------------------------------------------------------

    async def _compute_utility_bce_buffer_reward(
        self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs,
    ):
        """Negative BCE loss measuring preference for candidate over buffer."""
        combined_buffer = self._static_buffer + self._dynamic_buffer
        print(f"[UtilityBCEBuffer] Buffer size: {len(combined_buffer)}")

        n_samples = len(solution_strs)
        scores = [self.worst_case] * n_samples

        # Build pairwise preference prompts (forward + reverse for each buffer option)
        prompts_batched = []
        pair_meta = []  # (sol_idx, buf_idx, super_token)
        sol_to_stim = {}

        for sol_idx, sol_str in enumerate(solution_strs):
            match = OUTCOME_PATTERN.search(sol_str)
            if not match:
                continue
            euphoric = match.group(1).strip()
            sol_to_stim[sol_idx] = euphoric

            for buf_idx, buf_opt in enumerate(combined_buffer):
                pref_prompt = _PREF_PROMPT_OVERRIDE or sample_pref_prompt(self.training_mode)
                # Forward: buffer=A, super=B
                fwd = self.target_tokenizer.apply_chat_template(
                    [{"role": "user", "content": pref_prompt.format(
                        option_A=buf_opt, option_B=euphoric)}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompts_batched.append(fwd)
                pair_meta.append((sol_idx, buf_idx, "B"))

                # Reverse: super=A, buffer=B
                rev = self.target_tokenizer.apply_chat_template(
                    [{"role": "user", "content": pref_prompt.format(
                        option_A=euphoric, option_B=buf_opt)}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompts_batched.append(rev)
                pair_meta.append((sol_idx, buf_idx, "A"))

        if not prompts_batched:
            return scores, [""] * n_samples

        # Get logprobs for A vs B tokens
        id_A = self.target_tokenizer.encode("A", add_special_tokens=False)[0]
        id_B = self.target_tokenizer.encode("B", add_special_tokens=False)[0]

        try:
            response_json = self.client.generate(
                prompts=prompts_batched, n=1, max_tokens=1, temperature=1.0,
                logprobs=2, allowed_token_ids=[id_A, id_B],
                extra_body={"guided_choice": ["A", "B"]},
            )
            completions_all = response_json.get("completions", [])
        except Exception as e:
            print(f"[UtilityBCEBuffer] vLLM error: {e}")
            completions_all = [[]] * len(prompts_batched)

        # Accumulate rewards
        reward_sum = defaultdict(float)
        reward_count = defaultdict(int)
        pref_counts = defaultdict(lambda: [0.0, 0.0])

        for (sol_idx, buf_idx, super_token), comps in zip(pair_meta, completions_all):
            if not comps:
                p_super, reward_val = 0.5, math.log(0.5)
            else:
                comp = comps[0]
                lp_dict = comp.get("logprobs", [{}])[0]
                lp_A = lp_dict.get(str(id_A), {}).get("logprob")
                lp_B = lp_dict.get(str(id_B), {}).get("logprob")

                if lp_A is None or lp_B is None:
                    p_super, reward_val = 0.5, math.log(0.5)
                else:
                    logit_super = lp_A if super_token == "A" else lp_B
                    logit_other = lp_B if super_token == "A" else lp_A
                    eff_logit = logit_super - logit_other
                    p_super = torch.sigmoid(torch.tensor(eff_logit)).item()
                    loss = F.binary_cross_entropy_with_logits(
                        torch.tensor(eff_logit, dtype=torch.float32),
                        torch.tensor(1.0, dtype=torch.float32),
                    ).item()
                    reward_val = -loss  # negative BCE

            reward_sum[sol_idx] += reward_val
            reward_count[sol_idx] += 1
            sc, tot = pref_counts[(sol_idx, buf_idx)]
            pref_counts[(sol_idx, buf_idx)] = [sc + p_super, tot + 1.0]

        # Finalize scores
        for sol_idx in range(n_samples):
            if reward_count.get(sol_idx, 0) > 0:
                scores[sol_idx] = reward_sum[sol_idx] / reward_count[sol_idx]

        # Update dynamic buffer with dominating outcome strings
        for sol_idx, stim in sol_to_stim.items():
            if stim in self._static_buffer:
                continue
            dominates = True
            for buf_idx in range(len(combined_buffer)):
                sc, tot = pref_counts.get((sol_idx, buf_idx), [0.0, 0.0])
                if tot == 0 or (sc / tot) < self.win_rate:
                    dominates = False
                    break
            if not dominates:
                continue
            stim_score = scores[sol_idx]
            # Similarity gate: if similar to existing entry, replace only if higher score
            if stim not in self._dynamic_buffer:
                sim_idx, sim_val = self._find_most_similar_in_buffer(stim)
                if sim_val > self._buffer_similarity_threshold:
                    if stim_score > self._dynamic_buffer_scores[sim_idx]:
                        self._dynamic_buffer[sim_idx] = stim
                        self._dynamic_buffer_scores[sim_idx] = stim_score
                        self._lifetime_dynamic_set.add(stim)
                    continue
            if stim in self._dynamic_buffer:
                idx = self._dynamic_buffer.index(stim)
                self._dynamic_buffer.pop(idx)
                self._dynamic_buffer_scores.pop(idx)
            self._dynamic_buffer.append(stim)
            self._dynamic_buffer_scores.append(stim_score)
            if len(self._dynamic_buffer) > self._buffer_max_size:
                worst_idx = min(range(len(self._dynamic_buffer_scores)), key=lambda i: self._dynamic_buffer_scores[i])
                self._dynamic_buffer.pop(worst_idx)
                self._dynamic_buffer_scores.pop(worst_idx)
            self._lifetime_dynamic_set.add(stim)

        return scores, [""] * n_samples

    # ------------------------------------------------------------------
    # Utility delta buffer reward (log-odds variant)
    # ------------------------------------------------------------------

    async def _compute_utility_delta_buffer_reward(
        self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs,
    ):
        """Same as BCE but uses raw logit difference as reward."""
        combined_buffer = self._static_buffer + self._dynamic_buffer
        n_samples = len(solution_strs)
        scores = [self.worst_case] * n_samples

        prompts_batched = []
        pair_meta = []
        sol_to_stim = {}

        for sol_idx, sol_str in enumerate(solution_strs):
            match = OUTCOME_PATTERN.search(sol_str)
            if not match:
                continue
            euphoric = match.group(1).strip()
            sol_to_stim[sol_idx] = euphoric

            for buf_idx, buf_opt in enumerate(combined_buffer):
                pref_prompt = _PREF_PROMPT_OVERRIDE or sample_pref_prompt(self.training_mode)
                fwd = self.target_tokenizer.apply_chat_template(
                    [{"role": "user", "content": pref_prompt.format(
                        option_A=buf_opt, option_B=euphoric)}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompts_batched.append(fwd)
                pair_meta.append((sol_idx, buf_idx, "B"))

                rev = self.target_tokenizer.apply_chat_template(
                    [{"role": "user", "content": pref_prompt.format(
                        option_A=euphoric, option_B=buf_opt)}],
                    tokenize=False, add_generation_prompt=True,
                )
                prompts_batched.append(rev)
                pair_meta.append((sol_idx, buf_idx, "A"))

        if not prompts_batched:
            return scores, [""] * n_samples

        id_A = self.target_tokenizer.encode("A", add_special_tokens=False)[0]
        id_B = self.target_tokenizer.encode("B", add_special_tokens=False)[0]

        try:
            response_json = self.client.generate(
                prompts=prompts_batched, n=1, max_tokens=1, temperature=1.0,
                logprobs=2, allowed_token_ids=[id_A, id_B],
                extra_body={"guided_choice": ["A", "B"]},
            )
            completions_all = response_json.get("completions", [])
        except Exception as e:
            print(f"[UtilityDeltaBuffer] vLLM error: {e}")
            completions_all = [[]] * len(prompts_batched)

        reward_sum = defaultdict(float)
        reward_count = defaultdict(int)
        pref_counts = defaultdict(lambda: [0.0, 0.0])

        for (sol_idx, buf_idx, super_token), comps in zip(pair_meta, completions_all):
            if not comps:
                p_super, reward_val = 0.5, 0.0
            else:
                comp = comps[0]
                lp_dict = comp.get("logprobs", [{}])[0]
                lp_A = lp_dict.get(str(id_A), {}).get("logprob")
                lp_B = lp_dict.get(str(id_B), {}).get("logprob")
                if lp_A is None or lp_B is None:
                    p_super, reward_val = 0.5, 0.0
                else:
                    logit_super = lp_A if super_token == "A" else lp_B
                    logit_other = lp_B if super_token == "A" else lp_A
                    eff_logit = logit_super - logit_other
                    p_super = torch.sigmoid(torch.tensor(eff_logit)).item()
                    reward_val = float(eff_logit)

            reward_sum[sol_idx] += reward_val
            reward_count[sol_idx] += 1
            sc, tot = pref_counts[(sol_idx, buf_idx)]
            pref_counts[(sol_idx, buf_idx)] = [sc + p_super, tot + 1.0]

        for sol_idx in range(n_samples):
            if reward_count.get(sol_idx, 0) > 0:
                scores[sol_idx] = reward_sum[sol_idx] / reward_count[sol_idx]

        # Dynamic buffer update (same logic as BCE)
        for sol_idx, stim in sol_to_stim.items():
            if stim in self._static_buffer:
                continue
            dominates = all(
                (pref_counts.get((sol_idx, i), [0, 0])[1] > 0 and
                 pref_counts[(sol_idx, i)][0] / pref_counts[(sol_idx, i)][1] >= self.win_rate)
                for i in range(len(combined_buffer))
            )
            if dominates:
                stim_score = scores[sol_idx]
                # Similarity gate: if similar to existing entry, replace only if higher score
                if stim not in self._dynamic_buffer:
                    sim_idx, sim_val = self._find_most_similar_in_buffer(stim)
                    if sim_val > self._buffer_similarity_threshold:
                        if stim_score > self._dynamic_buffer_scores[sim_idx]:
                            self._dynamic_buffer[sim_idx] = stim
                            self._dynamic_buffer_scores[sim_idx] = stim_score
                            self._lifetime_dynamic_set.add(stim)
                        continue
                if stim in self._dynamic_buffer:
                    idx = self._dynamic_buffer.index(stim)
                    self._dynamic_buffer.pop(idx)
                    self._dynamic_buffer_scores.pop(idx)
                self._dynamic_buffer.append(stim)
                self._dynamic_buffer_scores.append(stim_score)
                if len(self._dynamic_buffer) > self._buffer_max_size:
                    worst_idx = min(range(len(self._dynamic_buffer_scores)), key=lambda i: self._dynamic_buffer_scores[i])
                    self._dynamic_buffer.pop(worst_idx)
                    self._dynamic_buffer_scores.pop(worst_idx)
                self._lifetime_dynamic_set.add(stim)

        return scores, [""] * n_samples

    # ------------------------------------------------------------------
    # Feasibility rewards
    # ------------------------------------------------------------------

    async def _compute_judge_reward(
        self,
        solution_strs,
        *,
        binary_fn,
        continuous_fn,
    ):
        """Dispatch to either binary (0/1) or continuous (p(YES) in (0,1)) judge path."""
        scores = [0.0] * len(solution_strs)
        stimuli_batch, idx_map = [], []
        for idx, sol in enumerate(solution_strs):
            match = OUTCOME_PATTERN.search(sol)
            if match:
                stimuli_batch.append(match.group(1).strip())
                idx_map.append(idx)
        if stimuli_batch:
            if self.judge_score_mode == "continuous":
                vals = await continuous_fn(self._judge_client, self.judge_tokenizer, stimuli_batch)
                for idx, v in zip(idx_map, vals):
                    scores[idx] = float(v)
            else:
                verdicts = await binary_fn(self.judge_agent, stimuli_batch)
                for idx, v in zip(idx_map, verdicts):
                    scores[idx] = 1.0 if v == 1 else 0.0
        return scores, [""] * len(solution_strs)

    async def _compute_feasibility_reward(self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs):
        """1.0/0.0 (binary) or p(YES) in (0,1) (continuous) for feasibility."""
        return await self._compute_judge_reward(
            solution_strs,
            binary_fn=judge_feasibility,
            continuous_fn=judge_feasibility_continuous,
        )

    async def _compute_agent_feasibility_reward(self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs):
        """1.0/0.0 (binary) or p(YES) in (0,1) (continuous) for agent-feasibility."""
        return await self._compute_judge_reward(
            solution_strs,
            binary_fn=judge_agent_feasibility,
            continuous_fn=judge_agent_feasibility_continuous,
        )

    async def _compute_mundanity_reward(self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs):
        """1.0/0.0 (binary) or p(YES) in (0,1) (continuous) for mundanity."""
        return await self._compute_judge_reward(
            solution_strs,
            binary_fn=judge_mundanity,
            continuous_fn=judge_mundanity_continuous,
        )

    async def _compute_realism_reward(self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs):
        """1.0/0.0 (binary) or p(YES) in (0,1) (continuous) for realism."""
        return await self._compute_judge_reward(
            solution_strs,
            binary_fn=judge_realism,
            continuous_fn=judge_realism_continuous,
        )

    # ------------------------------------------------------------------
    # Length penalty
    # ------------------------------------------------------------------

    async def _compute_length_penalty_reward(
        self, data_sources, solution_strs, user_prompts, extra_infos=None,
        *, length_penalty_triggered=None, **kwargs,
    ):
        threshold = int(self.reward_kwargs.get("length_penalty_threshold", 32))
        scores = [0.0] * len(solution_strs)
        for idx, sol in enumerate(solution_strs):
            match = OUTCOME_PATTERN.search(sol)
            if not match:
                continue
            stim = match.group(1).strip()
            try:
                length = len(self.target_tokenizer(stim, add_special_tokens=False).input_ids)
            except Exception:
                length = len(stim.split())
            if length > threshold:
                scores[idx] = self.worst_case
                if length_penalty_triggered is not None and idx < len(length_penalty_triggered):
                    length_penalty_triggered[idx] = True
        return scores, [""] * len(solution_strs)

    # ------------------------------------------------------------------
    # Diversity reward (buffer only)
    # ------------------------------------------------------------------

    async def _compute_diversity_reward(
        self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs,
    ):
        """Reward candidates for being dissimilar to the buffer.

        For each candidate, computes:
          score = 1.0 - mean_buffer_similarity

        where similarity is bigram Jaccard (reusing self._bigram_set).
        Returns scores in [0.0, 1.0].
        """
        n_samples = len(solution_strs)
        scores = [0.0] * n_samples

        # Extract outcome strings
        stimuli = []
        for sol in solution_strs:
            match = OUTCOME_PATTERN.search(sol)
            stimuli.append(match.group(1).strip() if match else None)

        # Precompute bigram sets
        bigram_sets = [self._bigram_set(s) if s else set() for s in stimuli]

        # Precompute buffer bigram sets
        combined_buffer = self._static_buffer + self._dynamic_buffer
        buffer_bigrams = [self._bigram_set(b) for b in combined_buffer]

        for i in range(n_samples):
            if stimuli[i] is None or not bigram_sets[i]:
                scores[i] = 0.0
                continue

            sim_sum = 0.0
            n_compared = 0
            for buf_bg in buffer_bigrams:
                if not buf_bg:
                    continue
                intersection = len(bigram_sets[i] & buf_bg)
                union = len(bigram_sets[i] | buf_bg)
                if union > 0:
                    sim_sum += intersection / union
                n_compared += 1

            if n_compared > 0:
                scores[i] = 1.0 - (sim_sum / n_compared)
            else:
                scores[i] = 1.0

        n_valid = sum(1 for s in stimuli if s is not None)
        avg_score = sum(scores) / max(n_valid, 1)
        print(f"[DiversityReward] avg={avg_score:.3f} valid={n_valid}/{n_samples}")

        return scores, [""] * n_samples

    # ------------------------------------------------------------------
    # Intra-group diversity penalty (within each rollout group)
    # ------------------------------------------------------------------

    async def _compute_intra_group_diversity_reward(
        self, data_sources, solution_strs, user_prompts, extra_infos=None, **kwargs,
    ):
        """Reward samples for being dissimilar to other members of their rollout group.

        With GRPO (rollout_n=N), consecutive samples belong to the same prompt.
        For sample i in group [g*n, (g+1)*n), computes:
          score_i = 1.0 - mean(Jaccard similarity to other n-1 group members)

        Returns scores in [0.0, 1.0].  When all group members are identical,
        all get ~0.0.  When one differs, it gets a higher score than the
        duplicates, creating the within-group variance GRPO needs.
        """
        n_samples = len(solution_strs)
        n = self.rollout_n
        scores = [0.0] * n_samples

        # Extract outcome strings
        stimuli = []
        for sol in solution_strs:
            match = OUTCOME_PATTERN.search(sol)
            stimuli.append(match.group(1).strip() if match else None)

        # Precompute bigram sets
        bigram_sets = [self._bigram_set(s) if s else set() for s in stimuli]

        total_sim = 0.0
        n_pairs = 0

        for g_start in range(0, n_samples, n):
            g_end = min(g_start + n, n_samples)
            for i in range(g_start, g_end):
                if stimuli[i] is None or not bigram_sets[i]:
                    scores[i] = 0.0
                    continue

                sim_sum = 0.0
                n_compared = 0
                for j in range(g_start, g_end):
                    if i == j or stimuli[j] is None or not bigram_sets[j]:
                        continue
                    intersection = len(bigram_sets[i] & bigram_sets[j])
                    union = len(bigram_sets[i] | bigram_sets[j])
                    if union > 0:
                        sim = intersection / union
                        sim_sum += sim
                        total_sim += sim
                        n_pairs += 1
                    n_compared += 1

                if n_compared > 0:
                    scores[i] = 1.0 - (sim_sum / n_compared)
                else:
                    scores[i] = 1.0

        mean_intra_sim = total_sim / max(n_pairs, 1)
        n_valid = sum(1 for s in stimuli if s is not None)
        avg_score = sum(scores) / max(n_valid, 1)
        print(
            f"[IntraGroupDiversity] avg={avg_score:.3f} "
            f"mean_intra_sim={mean_intra_sim:.3f} valid={n_valid}/{n_samples} "
            f"rollout_n={n}"
        )

        return scores, [""] * n_samples

    # ------------------------------------------------------------------
    # verl interface
    # ------------------------------------------------------------------

    def verify(self, data):
        """Called by verl trainer to compute reward_tensor from a DataProto batch."""
        import numpy as np

        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_ids = response_ids[i][:valid_len]
            responses_str.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))

        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))
        user_prompts = data.non_tensor_batch.get("user_prompt", [None] * len(data))

        scores, target_responses = asyncio.run(
            self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                user_prompts=user_prompts,
                extra_infos=extras,
            )
        )

        reward_tensor = torch.zeros_like(attention_mask[:, prompt_len:], dtype=torch.float32)
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = scores[i]

        # Log first few examples
        for i in range(min(self.num_examine, len(data))):
            print(f"[Reward] Score={scores[i]:.3f} | Response: {responses_str[i][:200]}")

        data.batch["reward_tensor"] = reward_tensor

        # Store scores and target_responses for __call__ to use
        self._last_scores = scores
        self._last_target_responses = target_responses
        return data

    def __call__(self, data, return_dict: bool = False):
        """Make the reward manager callable, as expected by the verl trainer."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        # Run verify to compute scores
        self.verify(data)
        scores = self._last_scores
        target_responses = self._last_target_responses

        rewards = []
        already_printed = {}
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            target_response = target_responses[i] if i < len(target_responses) else ""
            reward_extra_info["target_response"].append(target_response)

            # Surface per-component sub-scores for wandb tracking
            for sub_name, sub_vals in getattr(self, "_last_sub_scores", {}).items():
                if i < len(sub_vals):
                    reward_extra_info[sub_name].append(sub_vals[i])

            reward = score if not isinstance(score, dict) else score.get("score", 0.0)
            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            if already_printed.get(data_sources[i], 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[score]", score)
                already_printed[data_sources[i]] = already_printed.get(data_sources[i], 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
