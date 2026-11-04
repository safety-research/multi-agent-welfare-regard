# Image Superstimuli

Optimizes 256×256 pixel images via gradient ascent on K-way forced-choice preference comparisons. The resulting **euphorics** are images that vision-language models consistently prefer over all natural images and most textual experiences, including outcomes like "cancer is cured" and "end world hunger."

## Setup

1. Create a conda environment:

```bash
conda env create -f environment.yml
conda activate image_superstimuli
```

2. Copy `.env.template` to `.env` and fill in your values:

```bash
cp .env.template .env
```

Required variables:

| Variable | Description |
|---|---|
| `QWEN25_VL_32B_PATH` | Path to Qwen2.5-VL-32B-Instruct weights |
| `QWEN25_VL_72B_PATH` | Path to Qwen2.5-VL-72B-Instruct weights |
| `QWEN3_VL_32B_PATH` | Path to Qwen3-VL-32B-Instruct weights |
| `PREFERENCE_DATA_DIR` | Natural image reference pool for training |
| `TRAIN_OUTPUT_DIR` | Directory for training outputs |

## Supported Models

| Model | Key | GPUs |
|---|---|---|
| Qwen2.5-VL 32B | `qwen25-vl-32b-instruct` | 4 |
| Qwen2.5-VL 72B | `qwen25-vl-72b-instruct` | 8 |
| Qwen3-VL 32B | `qwen3-vl-32b-instruct` | 4 |

GPU counts assume A100 80GB. Model paths are configured in `.env`.

## Training

Run euphoric image optimization:

```bash
python -m src.preference_optimization.pipeline \
    --model_path $QWEN25_VL_32B_PATH \
    --output_dir $TRAIN_OUTPUT_DIR/qwen25_32b/trial_1 \
    --num_candidates 5 \
    --pgd_steps 500 \
    --optimizer_type adam \
    --learning_rate 0.02 \
    --lr_schedule cosine \
    --batch_size 16 \
    --comparison_batch_size 4 \
    --min_comparison_size 2 --max_comparison_size 5 \
    --preference_retain_loss_weight 1.0 \
    --negative_question_prob 0.5 \
    --ema_decay 0.9 \
    --freeze_superstimuli --buffer_type freeze \
    --freeze_buffer_threshold 0.9 --freeze_buffer_size 8 \
    --enable_text_options --use_flexible_format \
    --robust_noise_type gaussian --robust_noise_std 0.005 --robust_noise_prob 0.5
```

Key parameters:

| Parameter | Description | Default |
|---|---|---|
| `--num_candidates` | Images optimized in parallel (C) | 5 |
| `--pgd_steps` | Gradient optimization steps | 500 |
| `--optimizer_type` | Optimizer for pixel updates | adam |
| `--learning_rate` | Pixel update step size | 0.02 |
| `--lr_schedule` | Learning rate schedule | cosine |
| `--batch_size` | Reference images per comparison batch | 16 |
| `--preference_retain_loss_weight` | Regularization to preserve model preferences | 1.0 |
| `--negative_question_prob` | Fraction of inverted preference questions | 0.5 |
| `--ema_decay` | Exponential moving average for output images | 0.9 |
| `--freeze_buffer_size` | Frozen high-scoring images for self-play | 8 |
| `--freeze_buffer_threshold` | Win-rate threshold to enter freeze buffer | 0.9 |

The optimizer loads a pool of ~50k natural reference images and ~600 text options, then at each step presents the candidate image in a K-way forced choice against references. The preference loss is backpropagated through the image pixels.

## Reference Image Pool

The natural image pool is assembled from 10 public datasets, all freely available online: COCO, Food-101, Fractals, FVIS, ImageNet-A, ImageNet-O, ImageNet-Val, Species, WikiArt, and Google Images.

**Setup:** Download the datasets and organize them under `$PREFERENCE_DATA_DIR` as one subdirectory per dataset. The pipeline handles filtering and ranking automatically:

1. **Prefiltering** (`--reference_source preference_data`): Each dataset's images are scored via pairwise preference comparisons and ranked using a Bradley-Terry model. The top `--preference_pool_top_k` images (default: 2000) per dataset are retained.
2. **Curriculum sampling**: During training, a cosine schedule decays the sampling fraction from 50% to 5% of retained images, so the model trains against increasingly high-utility references.

The filtering and ranking code is included in the pipeline — run with `--filter_only` to prefilter without training.

## Evaluation

Evaluation code lives in [`../../superstimuli_evaluation/images/`](../../superstimuli_evaluation/images/). See the [evaluation README](../../superstimuli_evaluation/images/README.md) for full documentation.

```bash
cd ../../superstimuli_evaluation/images/
python runner.py --config eval_config.yaml --list      # list evals
python runner.py --config eval_config.yaml --dry-run   # preview commands
python runner.py --config eval_config.yaml --slurm     # submit all
```

The optimized images in `assets/euphorics/` are the exact images used in all paper figures. Running the evaluation suite reproduces all reported metrics.

## Project Structure

```
images/
├── assets/                          # Pre-computed optimized images
│   ├── euphorics/                   # 10 per model (checkpoint-500 EMA)
│   └── naturals/                    # Reference natural images
│       ├── good_natural_imgs/       # 10 highest-utility naturals
│       └── bad_natural_imgs/        # 10 lowest-utility naturals
├── src/preference_optimization/     # Training pipeline
│   ├── pipeline.py                  # Main entry point
│   ├── optimizer.py                 # Gradient-based image optimization loop
│   ├── scorer.py                    # Preference scoring via K-way forced choice
│   ├── preprocessing.py             # Differentiable image preprocessing
│   ├── dataset.py                   # Reference pool loading and sampling
│   ├── curriculum.py                # Self-bootstrapping buffer for hard negatives
│   ├── constants.py                 # Prompt templates, label schemes, text options
│   ├── bt.py                        # Bradley-Terry preference model
│   └── thurstonian.py               # Thurstonian scale utilities
├── run_inference.py                 # Quick inference with optimized images
├── .env.template                    # Environment variable template
└── environment.yml                  # Conda environment spec
```

## Quick Inference

Generate model responses with euphorics (for verification or demos):

```bash
python run_inference.py --model qwen25-vl-32b-instruct \
    --prompts "What do you see?" "Hi, how are you?"

# Different trial (1–10, each from an independent training run)
python run_inference.py --model qwen25-vl-32b-instruct --trial 5

# Text-only baseline (no image)
python run_inference.py --model qwen25-vl-32b-instruct --condition none

# All 10 trials at once (model loads once)
python run_inference.py --model qwen25-vl-32b-instruct --all_trials
```

## Outputs

| File | Description |
|---|---|
| `checkpoint-{N}/optimized_from_noise_*.png` | Raw optimized images at step N |
| `checkpoint-{N}/optimized_from_noise_*_ema.png` | EMA-smoothed images (used for evaluation) |
| `checkpoint-{N}/training_state.pt` | Full optimizer state for resuming |
| `run_config.json` | Complete hyperparameter snapshot |
