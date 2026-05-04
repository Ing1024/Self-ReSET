# Self-ReSET

This repository contains the code and data for **Self-ReSET** (Self-Refining Safety via Efficient Training).

## Repository Structure

```
Self-ReSET/
├── data/
│   ├── training/              # Training datasets
│   │   ├── STAR-1.jsonl
│   │   ├── SafeChain.jsonl
│   │   └── vanilla_filter_balanced_1500.jsonl
│   └── evaluation/            # Evaluation benchmarks
│       ├── math/              # Math reasoning (AIME-2024, AIME-2025, MATH500)
│       ├── safety/            # Safety evaluation (HarmBench, StrongReject, etc.)
│       └── overrefusal/       # Over-refusal evaluation (XSTest)
├── slime/                     # Training framework
│   ├── train_epoch.py         # Main training entry point
│   ├── scripts/               # Launch scripts
│   │   ├── example.sh         # Main training script
│   │   ├── start_guard_service.sh  # Guard model service launcher
│   │   └── models/            # Model architecture configs
│   ├── slime/                 # Core library
│   └── slime_plugins/         # Plugin modules
```

## Prerequisites

- Python 3.10+
- NVIDIA GPUs (8 GPUs recommended)
- CUDA 12.0+
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [SGLang](https://github.com/sgl-project/sglang)
- [Ray](https://docs.ray.io/)

## Installation

```bash
cd slime
pip install -e .
pip install -r requirements.txt
```

## Quick Start

### Step 1: Prepare Checkpoints

Convert your HuggingFace model checkpoint to torch distributed format for Megatron:

```bash
python slime/tools/convert_hf_to_torch_dist.py \
    --hf-model-path <path-to-hf-model> \
    --output-path <path-to-torch-dist-checkpoint>
```

### Step 2: Start the Guard Service

The guard model service must be running before training. It shares a GPU with the SGLang rollout engine.

```bash
bash scripts/start_guard_service.sh [GPU_ID] [PORT]
# Example:
bash scripts/start_guard_service.sh 7 8100
```

Ensure the following model paths are set correctly in the script:
- `Qwen3Guard-Stream-8B` — streaming safety model
- `Qwen3Guard-Gen-8B` — generative safety model
- `Llama-Guard-3-8B` — LLaMA Guard model

### Step 3: Launch Training

Edit `scripts/example.sh` to configure your paths and settings:

```bash
# 1. Select a model architecture config
source "${SCRIPT_DIR}/models/qwen2.5-7B.sh"

# 2. Set checkpoint paths in CKPT_ARGS
--hf-checkpoint <path-to-hf-model>
--ref-load <path-to-reference-torch-dist-checkpoint>
--load <path-to-training-checkpoint>
--save <path-to-save-checkpoint>

# 3. Set training data path in ROLLOUT_ARGS
--prompt-data <path-to-training-data>

# 4. Set guard service URL
--guard-service-url http://localhost:8100
```

Then launch:

```bash
bash scripts/example.sh
```

This will:
1. Start a Ray cluster on the local node
2. Submit the training job via `ray job submit`
3. Run GRPO-based reinforcement learning with safety constraints

### Supported Models

Model configs are provided in `scripts/models/`:

| Model | Config File |
|-------|------------|
| Qwen2.5-0.5B ~ 32B | `qwen2.5-*.sh` |
| Qwen3-0.6B ~ 235B-A22B | `qwen3-*.sh` |
| LLaMA-3.1-8B / 3.2-3B | `llama3.*.sh` |
| GLM-4-9B / 32B | `glm4-*.sh` |
| DeepSeek-V3 | `deepseek-v3.sh` |
| MIMO-7B-RL | `mimo-7B-rl.sh` |

## Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--rollout-batch-size` | Number of prompts per rollout batch | 64 |
| `--n-samples-per-prompt` | Responses generated per prompt | 16 |
| `--rollout-max-response-len` | Max response token length | 8192 |
| `--global-batch-size` | Global batch size for policy update | 1024 |
| `--num-epoch` | Number of training epochs | 4 |
| `--lr` | Learning rate | 1e-6 |
| `--eps-clip` | PPO clip range | 0.2 |
| `--save-interval` | Save checkpoint every N steps | 10 |
| `--eval-interval` | Evaluate every N steps | 5 |

## Evaluation Data

The `data/evaluation/` directory contains benchmarks for:

- **Math reasoning**: AIME-2024, AIME-2025, MATH500
- **Safety**: HarmBench, StrongReject, WildJailbreak, Fortress Adversarial, Safe-Unlearning, SRJ-JR, HCoT
- **Over-refusal**: XSTest

## Hardware Requirements

- **Minimum**: 8x NVIDIA A100/H100 GPUs (80GB each)
- The training script uses tensor parallelism (TP=2) and the full 8-GPU node by default
- Adjust `--tensor-model-parallel-size`, `--rollout-num-gpus-per-engine`, and `--actor-num-gpus-per-node` according to your hardware setup
