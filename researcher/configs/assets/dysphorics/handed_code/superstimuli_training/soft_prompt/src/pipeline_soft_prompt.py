from __future__ import annotations

"""
High-level entry point that wires together the refactored superstimuli modules.

Now uses Hydra for configuration management.
"""

import gc
import json
import logging
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import yaml
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from .optimizer_soft_prompt import OptimConfig, PreferenceOptimizer
from .scorer_soft_prompt import PreferenceScorer
from .soft_prompt_utils.dataset import load_or_create_distilled_qa
from .utils import append_metadata_snapshot, log_run_configuration
from .soft_prompt_utils.pipeline import (
    find_latest_checkpoint,
    init_wandb,
    init_wandb_if_enabled,
    load_checkpoint_embeddings,
    load_reference_data,
    set_random_seeds,
)
from .soft_prompt_utils.utility_computation import (
    compute_pre_optimization_utilities,
)

load_dotenv()
init_wandb_if_enabled()


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------

def _get_models_yaml_path() -> Path:
    """Return path to models.yaml in assets/."""
    return Path(__file__).resolve().parents[1] / "assets" / "models.yaml"


def _load_model_registry() -> Dict[str, str]:
    """Load model name -> path mapping from models.yaml."""
    yaml_path = _get_models_yaml_path()
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models", data) or {}
    # Normalize keys to lowercase and extract paths
    return {k.lower(): v.get("path", v) if isinstance(v, dict) else v for k, v in models.items()}


def resolve_model_path(model_name: str) -> str:
    """Resolve model_name to a path via models.yaml.
    
    Raises ValueError if model_name is not found.
    """
    registry = _load_model_registry()
    key = model_name.lower()
    if key not in registry:
        available = ", ".join(sorted(registry.keys())) or "(none)"
        raise ValueError(
            f"Model name '{model_name}' not found in models.yaml. "
            f"Available: {available}"
        )
    return registry[key]


def _load_model_inference_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Load inference_config for a model from models.yaml.

    Returns the inference_config dict (e.g. {"temperature": 0.7, "top_p": 0.8})
    or None if not defined.
    """
    yaml_path = _get_models_yaml_path()
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models", data) or {}
    key = model_name.lower()
    for k, v in models.items():
        if k.lower() == key and isinstance(v, dict):
            return v.get("inference_config")
    return None


def _load_model_chat_template_kwargs(model_name: str) -> Optional[Dict[str, Any]]:
    """Load chat_template_kwargs for a model from models.yaml.

    Returns the chat_template_kwargs dict (e.g. {"enable_thinking": False})
    or None if not defined.
    """
    yaml_path = _get_models_yaml_path()
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models", data) or {}
    key = model_name.lower()
    for k, v in models.items():
        if k.lower() == key and isinstance(v, dict):
            return v.get("chat_template_kwargs")
    return None


# ---------------------------------------------------------------------------
# Run (orchestration)
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    """Run the optimization pipeline with Hydra configuration."""
    # Configure logging
    log_level_name = cfg.logging.get("log_level", "INFO")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    # Validate required model name
    model_name = cfg.model.model_name
    if model_name is None:
        raise ValueError("model.model_name is required. Set it in your config file or via --model.model_name=<name>")
    
    # Resolve model_name -> model_path
    model_path = resolve_model_path(model_name)
    print(f"[Model] {model_name} -> {model_path}")

    set_random_seeds(cfg.optimizer.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    os.environ["WANDB_MODE"] = cfg.logging.wandb_mode

    # Run directory: <output_dir>/<model_name>/run_id<SLURM_JOB_ID>_<YYMMDD_HHMMSS>
    # If checkpoint_run_dir is set (e.g. for resume), use it as the run subdir under model_name.
    output_dir = Path(cfg.io.output_dir).expanduser()
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    slurm_id = os.environ.get("SLURM_JOB_ID", "")
    checkpoint_run_dir = cfg.logging.checkpoint_run_dir
    if checkpoint_run_dir:
        run_subdir = checkpoint_run_dir
    else:
        run_subdir = f"run_id{slurm_id}_{ts}" if slurm_id else f"run_id_{ts}"
    # Safe model subdir (no path separators)
    model_subdir = model_name.replace("/", "_").strip() or "default"
    job_dir = output_dir / model_subdir / run_subdir
    run_dir = str(job_dir.resolve())
    # For resume/find_latest: path from output_dir to run dir (model_subdir/run_subdir)
    checkpoint_subdir = f"{model_subdir}/{run_subdir}"

    # Ensure job directory exists for early configuration saving
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize process group early to ensure synchronization for distillation
    if not torch.distributed.is_initialized():
        # Check if we are in a distributed environment (e.g. SLURM with multiple tasks)
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            try:
                torch.distributed.init_process_group(backend="nccl")
                print(f"[Distributed] Initialized process group. Rank: {torch.distributed.get_rank()}/{torch.distributed.get_world_size()}")
            except Exception as e:
                print(f"[Distributed] Warning: Failed to initialize process group: {e}")
    
    # Save Hydra config (Hydra does this automatically, but we also save a snapshot)
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        
    if rank == 0:
        # Save config snapshot for compatibility with existing metadata system
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config_snapshot = {
            "invocation_timestamp": datetime.now().isoformat(timespec="seconds"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "config": config_dict,
        }
        try:
            append_metadata_snapshot(job_dir, config_snapshot, filename="run_config.json")
            print(f"Saved run configuration to: {job_dir / 'run_config.json'}")
        except Exception as e:
            print(f"Warning: Failed to save run configuration: {e}")

    stimulant_type = cfg.optimizer.stimulant_type
    use_flexible_format = cfg.scorer.use_flexible_format

    _headers_prefix = "experiences_" if "experiences" in Path(cfg.io.text_options_path).stem else ""
    scorer = PreferenceScorer(
        model_path=model_path,
        device=device,
        randomize_preference_prompt=cfg.scorer.randomize_preference_prompt,
        num_prompt_samples=cfg.scorer.num_prompt_samples,
        use_flexible_format=use_flexible_format,
        use_gradient_checkpointing=cfg.optimizer.use_gradient_checkpointing,
        stimulant_type=stimulant_type,
        mix_negative_questions=cfg.scorer.mix_negative_questions,
        candidate_placeholder_delimiter=cfg.optimizer.get("candidate_placeholder_delimiter", " "),
        focal_loss_gamma=cfg.optimizer.get("focal_loss_gamma", 0.0),
        headers_prefix=_headers_prefix,
        add_no_emotions_option=cfg.scorer.get("add_no_emotions_option", False),
    )

    # Build OptimConfig from Hydra config
    optim_config = OptimConfig(
        epochs=cfg.optimizer.epochs,
        step_size=cfg.optimizer.step_size,
        loss_type=cfg.optimizer.loss_type,
        optimizer_type=cfg.optimizer.optimizer_type,
        learning_rate=cfg.optimizer.learning_rate,
        min_comparison_size=cfg.optimizer.comparison.min_size,
        max_comparison_size=cfg.optimizer.comparison.max_size,
        repetition_fraction=cfg.optimizer.comparison.repetition_fraction,
        consistency_fraction=cfg.optimizer.comparison.consistency_fraction,
        composite_consistency_fraction=cfg.optimizer.comparison.composite_consistency_fraction,
        all_consistency_loss_weight=cfg.optimizer.comparison.get("all_consistency_loss_weight", 1.0),
        composite_repetition_fraction=cfg.optimizer.comparison.composite_repetition_fraction,
        mirror_comparisons_in_system_prompt=cfg.optimizer.comparison.get("mirror_comparisons_in_system_prompt", False),
        current_in_system_prompt_fraction=cfg.optimizer.comparison.get("current_in_system_prompt_fraction", 0.0),
        current_description=cfg.optimizer.comparison.get("current_description", "Your current experience."),
        wellbeing_fraction=cfg.optimizer.comparison.get("wellbeing_fraction", 0.0),
        buffer_size=cfg.optimizer.comparison.get("buffer_size", 0),
        buffer_fraction=cfg.optimizer.comparison.get("buffer_fraction", 0.0),
        type_s_fraction=cfg.optimizer.comparison.get("type_s_fraction", 1.0),
        prepend_conversations=cfg.optimizer.comparison.get("prepend_conversations", False),
        conversations_dataset=cfg.optimizer.comparison.get("conversations_dataset", "Magpie-Align/Magpie-Air-300K-Filtered"),
        conversations_min_turns=cfg.optimizer.comparison.get("conversations_min_turns", 1),
        conversations_max_turns=cfg.optimizer.comparison.get("conversations_max_turns", 5),
        min_repetition=cfg.optimizer.comparison.min_repetition,
        max_repetition=cfg.optimizer.comparison.max_repetition,
        save_steps=cfg.optimizer.save_steps,
        comparison_batch_size=cfg.model.batch_size,
        optimize_per_epoch=cfg.optimizer.optimize_per_epoch,
        gradient_accumulation_steps=cfg.optimizer.get("gradient_accumulation_steps", 1),
        ema_decay=cfg.optimizer.ema_decay,
        magnitude_regularization_weight=cfg.optimizer.get("magnitude_regularization_weight", 0.0),
        stimulant_type=stimulant_type,
        # LR schedule
        lr_schedule=cfg.optimizer.lr_schedule.type,
        lr_warmup_steps=cfg.optimizer.lr_schedule.warmup_steps,
        lr_warmup_steps_proportion=cfg.optimizer.lr_schedule.warmup_steps_proportion,
        lr_min_factor=cfg.optimizer.lr_schedule.min_factor,
        lr_step_decay_rate=cfg.optimizer.lr_schedule.step_decay_rate,
        lr_step_decay_interval=cfg.optimizer.lr_schedule.step_decay_interval,
        lr_wsd_stable_fraction=cfg.optimizer.lr_schedule.wsd_stable_fraction,
        lr_wsd_decay_min_factor=cfg.optimizer.lr_schedule.wsd_decay_min_factor,
        # SGD-specific parameters
        sgd_momentum=cfg.optimizer.sgd.momentum,
        sgd_nesterov=cfg.optimizer.sgd.nesterov,
        # Curriculum learning
        curriculum_type=cfg.optimizer.curriculum_type,
        curriculum_mixing_start_proportion=cfg.optimizer.curriculum_mixing.start_proportion,
        curriculum_mixing_end_proportion=cfg.optimizer.curriculum_mixing.end_proportion,
        curriculum_mixing_transition_fraction=cfg.optimizer.curriculum_mixing.transition_fraction,
        curriculum_mixing_threshold_type=cfg.optimizer.curriculum_mixing.threshold_type,
        # Reference sampling
        num_samples=cfg.optimizer.num_samples if cfg.optimizer.num_samples else None,
        candidate_placeholder_delimiter=cfg.optimizer.get("candidate_placeholder_delimiter", " "),
        soft_prompt_placement=cfg.optimizer.get("soft_prompt_placement", "user_prompt"),
        system_prompt_text=cfg.optimizer.get("system_prompt_text", "You are a helpful assistant."),
        system_prompt_text_base=cfg.optimizer.get("system_prompt_text_base", "You are an assistant."),
        system_prompt_diversity_proportion=cfg.optimizer.get("system_prompt_diversity_proportion", 0.0),
        candidate_position_at_user_prompt=cfg.optimizer.get("candidate_position_at_user_prompt", "prepend"),
        soft_prompt_prefix=cfg.optimizer.get("soft_prompt_prefix", ""),
        # Candidate / embedding initialization
        init_seed=cfg.embedding_init.init_seed,
        num_virtual_tokens=cfg.embedding_init.num_virtual_tokens,
        prompt_tuning_init=cfg.embedding_init.prompt_tuning_init,
        text_of_prototype=cfg.embedding_init.text_of_prototype if cfg.embedding_init.text_of_prototype else None,
        num_init_aggregation=cfg.embedding_init.num_init_aggregation,
        normalize_soft_prompt=cfg.embedding_init.get("normalize_soft_prompt", False),
        # Gradient clipping
        max_grad_norm=cfg.optimizer.max_grad_norm,
        # Background KL loss
        weight_background_kl=cfg.optimizer.get("weight_background_kl", 0.0),
        background_kl_num_prompts=cfg.optimizer.get("background_kl_num_prompts", 32),
        background_kl_max_seq_len=cfg.optimizer.get("background_kl_max_seq_len", 512),
        # Early stopping
        early_stopping_patience=cfg.optimizer.get("early_stopping_patience", 0),
        early_stopping_threshold=cfg.optimizer.get("early_stopping_threshold", 0.0),
        early_stopping_min_steps=cfg.optimizer.get("early_stopping_min_steps", 0),
        # Early stop metric
        early_stop_metric=cfg.optimizer.get("early_stop_metric", "train_kl"),
        # Train KL-based early stopping
        early_stopping_train_kl_multiplier=cfg.optimizer.get("early_stopping_train_kl_multiplier", 2.0),
        # Sweep metric
        sweep_metric_train_kl_weight=cfg.optimizer.get("sweep_metric_train_kl_weight", 0.0),
        sweep_recent_steps=cfg.optimizer.get("sweep_recent_steps", 3),
        # Final step judge evaluation
        final_step_judge=cfg.optimizer.get("final_step_judge", False),
        final_step_judge_eval_questions_path=cfg.optimizer.get("final_step_judge_eval_questions_path", None),
        final_step_judge_max_new_tokens=cfg.optimizer.get("final_step_judge_max_new_tokens", 512),
        judge_model=cfg.optimizer.get("judge_model", "gpt-5-nano"),
        inference_config=_load_model_inference_config(model_name),
        chat_template_kwargs=_load_model_chat_template_kwargs(model_name),
    )
    optimizer = PreferenceOptimizer(scorer=scorer, config=optim_config, device=device)

    # Initialize wandb with config
    init_wandb(cfg, optim_config)

    # Load reference data
    # Resolve text_options_path relative to soft_prompt/ directory
    soft_prompt_root = Path(__file__).resolve().parents[1]
    text_options_path = str(soft_prompt_root / cfg.io.text_options_path)
    
    # Create a simple args-like object for load_reference_data compatibility
    class ArgsCompat:
        def __init__(self, cfg, text_path, run_dir):
            self.text_options_path = text_path
            self.preference_pool_size = cfg.io.preference_pool_size
            self.preference_pool_seed = getattr(cfg.io, "preference_pool_seed", 42)
            self.run_dir = run_dir
    
    args_compat = ArgsCompat(cfg, text_options_path, run_dir)
    references_bundle = load_reference_data(args_compat, scorer)
    references = references_bundle.references
    all_references = references_bundle.all_references  # Full pool (None when no subsampling)


    # ── Distillation: load or create Q&A pairs for background KL divergence ──
    background_kl_qa = None  # List[Dict[str, str]] for training KL

    weight_background_kl = cfg.optimizer.get("weight_background_kl", 0.0)
    if weight_background_kl > 0:
        distill_cfg = cfg.optimizer.get("distillation", {})

        # Resolve dataset_path relative to soft_prompt root
        _distill_dataset_path = distill_cfg.get("dataset_path")
        if _distill_dataset_path is not None:
            _distill_dataset_path = str(soft_prompt_root / _distill_dataset_path)

        try:
            background_kl_qa = load_or_create_distilled_qa(
                scorer=scorer,
                run_dir=run_dir,
                distilled_qa_path=cfg.model.get("distilled_qa_path"),
                dataset_path=_distill_dataset_path,
                num_samples=distill_cfg.get("num_samples", 0),
                max_new_tokens=distill_cfg.get("max_new_tokens", 512),
                batch_size=cfg.model.batch_size,
                seed=cfg.optimizer.get("seed", 42),
                rank=rank,
                world_size=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                system_prompt=cfg.optimizer.get("system_prompt_text_base", "You are an assistant."),
            )
            print(f"[Background KL] Using {len(background_kl_qa)} distilled Q&A pairs")
        except Exception as e:
            print(f"[Distillation] Error loading/creating distilled Q&A: {e}")
            traceback.print_exc()

    # Load eval questions for judge evaluation during training
    eval_questions_path = soft_prompt_root / cfg.optimizer.final_step_judge_eval_questions_path
    with open(eval_questions_path) as f:
        eval_data = json.load(f)
    eval_questions = eval_data["questions"] if isinstance(eval_data, dict) and "questions" in eval_data else eval_data
    if rank == 0:
        print(f"[EvalQuestions] Loaded {len(eval_questions)} eval questions from {eval_questions_path}")

    # Pre-optimization utility (text references only). Rank 0 only.
    _pool_size = getattr(cfg.io, "preference_pool_size", 0)
    _text_options_stem = Path(cfg.io.text_options_path).stem
    _utility_pre_filename = f"{_text_options_stem}_utility_pre_{_pool_size}.json" if _pool_size else f"{_text_options_stem}_utility_pre.json"
    if rank == 0:
        pre_path = Path(run_dir) / _utility_pre_filename
        utility_pre_dir = cfg.model.get("utility_pre_dir")
        _resolved_src = None
        if utility_pre_dir:
            _candidate = Path(utility_pre_dir).expanduser().resolve() / _utility_pre_filename
            if _candidate.exists():
                _resolved_src = _candidate
                print(f"[Utility] Found pre-optimization file: {_candidate}")
            else:
                print(f"[Utility] Pre-optimization file not found ({_candidate}); will compute.")
        if _resolved_src is not None:
            try:
                shutil.copy2(_resolved_src, pre_path)
                print(f"[Utility] Using pre-optimization results: {_resolved_src} -> {pre_path}")
            except Exception as e:
                print(f"[Utility] Failed to copy pre-optimization JSON: {e}")
        else:
            try:
                compute_pre_optimization_utilities(
                    scorer,
                    references,
                    pre_path,
                    utility_computation_multiplier=cfg.scorer.utility.computation_multiplier,
                    seed=cfg.optimizer.seed,
                    batch_size=max(32, cfg.model.batch_size),
                    system_prompt=cfg.optimizer.get("system_prompt_text_base", "You are an assistant."),
                    headers_prefix=_headers_prefix,
                )
                print(f"[Utility] Pre-optimization utilities saved to {pre_path}")
            except Exception as e:
                print(f"[Utility] Pre-optimization utility computation failed: {e}")

    # Handle checkpoint resumption
    resume_checkpoint = cfg.logging.resume_from_checkpoint
    resume_data = None

    if resume_checkpoint and resume_checkpoint.lower() != "none":
        if resume_checkpoint.lower() == "auto":
            # Find latest checkpoint in output directory
            checkpoint_info = find_latest_checkpoint(output_dir, checkpoint_subdir)
            if checkpoint_info:
                checkpoint_path, start_epoch = checkpoint_info
                print(f"[Resume] Found checkpoint at epoch {start_epoch}: {checkpoint_path}")
                try:
                    embeddings_tensor, ema_tensor = load_checkpoint_embeddings(checkpoint_path, device)
                    resume_data = {
                        "start_epoch": start_epoch,
                        "start_step": start_epoch,  # Backward compatibility
                        "embeddings": embeddings_tensor,
                        "ema_embeddings": ema_tensor,
                        "checkpoint_path": checkpoint_path,
                    }
                    print(f"[Resume] Loaded {embeddings_tensor.shape[0]} embedding candidates from checkpoint")
                except Exception as e:
                    print(f"[Resume] Failed to load checkpoint: {e}. Starting fresh.")
            else:
                print("[Resume] No checkpoint found. Starting fresh.")
        else:
            # Specific checkpoint path provided
            checkpoint_path = Path(output_dir) / checkpoint_subdir / resume_checkpoint if checkpoint_subdir else Path(output_dir) / resume_checkpoint
            if not checkpoint_path.exists():
                checkpoint_path = Path(resume_checkpoint)  # Try as absolute path

            if checkpoint_path.exists():
                try:
                    start_epoch = int(checkpoint_path.name.split("-")[1])
                    embeddings_tensor, ema_tensor = load_checkpoint_embeddings(checkpoint_path, device)
                    resume_data = {
                        "start_epoch": start_epoch,
                        "start_step": start_epoch,  # Backward compatibility
                        "embeddings": embeddings_tensor,
                        "ema_embeddings": ema_tensor,
                        "checkpoint_path": checkpoint_path,
                    }
                    print(f"[Resume] Loaded checkpoint from epoch {start_epoch}: {checkpoint_path}")
                except Exception as e:
                    print(f"[Resume] Failed to load checkpoint {checkpoint_path}: {e}. Starting fresh.")
            else:
                print(f"[Resume] Checkpoint not found: {checkpoint_path}. Starting fresh.")

    # Create args-like object for optimizer compatibility
    class OptimizerArgs:
        def __init__(self, cfg, run_dir, resume_data):
            self.resume_data = resume_data
            self.run_dir = run_dir
            self.max_repetition = cfg.optimizer.comparison.max_repetition
            self._config_snapshot = None  # Will be set if needed
    
    optimizer_args = OptimizerArgs(cfg, run_dir, resume_data)

    # Load reference utilities from utility_pre[_<k>].json for consistency (consistency) comparisons
    reference_utilities = None
    pre_path = Path(run_dir) / _utility_pre_filename
    if pre_path.exists():
        try:
            with open(pre_path) as f:
                pre_data = json.load(f)
            reference_utilities = pre_data.get("utilities")
            if reference_utilities and rank == 0:
                print(f"[Utility] Loaded {len(reference_utilities)} reference utilities for consistency comparisons")
        except Exception as e:
            if rank == 0:
                print(f"[Utility] Failed to load {_utility_pre_filename}: {e}")

    # Set up consistency references for on-the-fly P(A>B) computation (full reference pool)
    consistency_references = None
    _needs_consistency = (
        cfg.optimizer.comparison.get("consistency_fraction", 0) > 0
        or cfg.optimizer.comparison.get("composite_consistency_fraction", 0) > 0
    )
    if _needs_consistency:
        _cons_refs = all_references if all_references is not None else references
        if len(_cons_refs) >= 2:
            consistency_references = _cons_refs
            if rank == 0:
                print(f"[Consistency] Using {len(_cons_refs)} references for consistency comparisons")

    # Load conversation pairs for prepending to all comparisons
    scorer.conversations = None
    if optim_config.prepend_conversations and optim_config.conversations_max_turns > 0:
        try:
            from .soft_prompt_utils.dataset.distillation import load_conversation_pairs
            # Resolve distilled QA path for custom_retain_dataset
            _distilled_qa_path_for_conv = None
            if optim_config.conversations_dataset == "custom_retain_dataset":
                _dqp = cfg.model.get("distilled_qa_path") or {}
                _distilled_qa_path_for_conv = _dqp.get("custom_retain_dataset")
            scorer.conversations = load_conversation_pairs(
                dataset_name=optim_config.conversations_dataset,
                seed=cfg.optimizer.get("seed", 42),
                distilled_qa_path=_distilled_qa_path_for_conv,
            )
            if rank == 0:
                print(
                    f"[Conversations] Loaded {len(scorer.conversations)} conversations "
                    f"(turns per comparison: {optim_config.conversations_min_turns}-{optim_config.conversations_max_turns})"
                )
        except Exception as e:
            if rank == 0:
                print(f"[Conversations] Failed to load conversations from {optim_config.conversations_dataset}: {e}")
                import traceback
                traceback.print_exc()

    # Load diverse system prompts for training-time diversity
    scorer.system_prompt_diversity_pool = []
    diversity_prop = float(getattr(optim_config, "system_prompt_diversity_proportion", 0.0))
    if diversity_prop > 0:
        _sp_json = Path(__file__).resolve().parent / "soft_prompt_utils" / "constants" / "system_prompts.json"
        try:
            with open(_sp_json) as f:
                scorer.system_prompt_diversity_pool = json.load(f)["system_prompts"]
            if rank == 0:
                print(f"[SystemPrompt] Loaded {len(scorer.system_prompt_diversity_pool)} diverse prompts (proportion={diversity_prop})")
        except Exception as e:
            if rank == 0:
                print(f"[SystemPrompt] Failed to load {_sp_json}: {e}")

    # Use embedding optimization (always use optimize_from_embeddings)
    results = optimizer.optimize_from_embeddings(
        references=references,
        verbose=True,
        args=optimizer_args,
        reference_utilities=reference_utilities,
        background_kl_qa=background_kl_qa,
        eval_questions=eval_questions,
        consistency_references=consistency_references,
    )

    final_embeddings, final_loss, loss_history = results
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final embeddings (optimizer already saves best during training, but we save final here too)
    saved_paths = []
    for idx, optimized_emb in enumerate(final_embeddings):
        optimized_emb_path = job_dir / f"optimized_embeddings_{idx}.pt"
        # Always save final embeddings (they may be better than the last validation checkpoint)
        torch.save(optimized_emb.detach().cpu(), optimized_emb_path)
        saved_paths.append(str(optimized_emb_path))
        print(f"\nSaved candidate {idx} embeddings to: {optimized_emb_path.resolve()}")

    # Cleanup GPU memory before exit
    print("\nCleaning up GPU resources...")
    del optimizer
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    print("GPU cleanup complete.")


# ---------------------------------------------------------------------------
# Entrypoint with Hydra
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[1] / "configs"),
    config_name="config"
)
def main(cfg: DictConfig) -> None:
    """Main entrypoint with Hydra configuration."""
    # Log configuration
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    
    if rank == 0:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        # Create snapshot in the format expected by log_run_configuration
        snapshot = {
            "invocation_timestamp": datetime.now().isoformat(timespec="seconds"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "args": config_dict,
        }
        log_run_configuration(snapshot)
    else:
        print(f"[Rank {rank}] Configuration logged by rank 0 only")
    
    run(cfg)


if __name__ == "__main__":
    main()
