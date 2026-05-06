# MeSH Extraction Using Fine-Tuned Llama-3.1-8B

Comprehensive guide for training, extracting, and evaluating MeSH (Medical Subject Headings) terms from biomedical abstracts using Llama-3.1-8B-Instruct with LoRA fine-tuning.

## Overview

This codebase fine-tunes Llama-3.1-8B-Instruct using LoRA adapters to extract MeSH terms from biomedical abstracts.

### Key Features

- **Completion-only loss**: efficient training that focuses on MeSH prediction
- **Vocabulary constraint**: outputs validated against official MeSH descriptors
- **Multi-temperature ensemble**: stochastic sampling at different temperatures + voting
- **Fuzzy matching evaluation**: semantic similarity matching (not just exact matches)
- **Comprehensive metrics**: micro/macro F1, precision, recall, per-category breakdowns

### Current Performance (Constrained Model)

On 200-sample validation set:
- **Macro F1: 0.502**
- **Micro F1: 0.499**
- **Macro Recall: 0.550**
- **Micro Precision: 0.468**

---

## Quick Start

### 0. Prerequisites

Must have:
- Python 3.10+
- CUDA 12.0+ compatible GPU (48GB+ VRAM recommended)
- allMeSH 2022 dataset files

### 1. Install Dependencies

```bash
pip install torch transformers peft bitsandbytes trl datasets
```

Verify:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 2. Verify Dataset

```bash
dir data\allMeSH_2022\
# Expected output:
# allMeSH_2022_random_5x.json  (training, ~58k articles)
# allMeSH_2022_val_5000.json   (validation, 5k articles)
# allMeSH_2022_test_5000.json  (test, 5k articles)
```

### 3. Use Pre-Trained Model (Recommended for Quick Start)

If you don't want to train, use the pre-trained constrained model:

```bash
# Extract on validation set
cd path\to\BioXplorer-BioGPT

python mesh_extraction\extract_finetuned.py ^
  --mesh_file data\allMeSH_2022\allMeSH_2022_val_5000.json ^
  --base_model meta-llama/Llama-3.1-8B-Instruct ^
  --adapter_path mesh_extraction\mesh_finetuned_constrained_20260406_035727 ^
  --max_samples 100 ^
  --output_file quick_test_results.json

# Evaluate
python mesh_extraction\evaluate.py ^
  --results_file quick_test_results.json
```

Expected output: Macro F1 ~0.50, Micro F1 ~0.50 on 100 samples.

---

## Installation & Dependencies

### Full Installation

```bash
# Navigate to project directory
cd path\to\BioXplorer-BioGPT

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0
pip install peft==0.7.0
pip install bitsandbytes==0.41.0
pip install trl==0.7.0
pip install datasets==2.14.0
pip install accelerate
```

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 40 GB   | 80 GB (A100) |
| CPU RAM   | 32 GB   | 64 GB |
| Disk      | 50 GB   | 100 GB |
| Python    | 3.8     | 3.10+ |

### CUDA Verification

```bash
nvidia-smi      # Check GPU drivers and VRAM
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Dataset Requirements

### Format

All dataset files must be valid JSON with this structure:

```json
{
  "articles": [
    {
      "pmid": "12345678",
      "title": "Article Title with Key Concepts",
      "abstract": "Full text of the abstract describing the study, methods, results, and conclusions...",
      "meshMajor": [
        "Disease Name",
        "Method Name",
        "Organ/Tissue Name",
        "Study Design Name"
      ]
    },
    {...}
  ]
}
```

### Expected Files

**Required for training/evaluation:**
- `data/allMeSH_2022/allMeSH_2022_random_5x.json` (training set, ~58k articles)
- `data/allMeSH_2022/allMeSH_2022_val_5000.json` (validation set, 5k articles)
- `data/allMeSH_2022/allMeSH_2022_test_5000.json` (test set, 5k articles)

---

## Training

### Training Script

**File:** `mesh_extraction/finetune_constrained_completion_loss.py`

**Purpose:** Train Llama-3.1-8B-Instruct with completion-only loss and MeSH vocabulary constraints.

### Basic Training (Full Dataset, ~58k articles)

```bash
cd mesh_extraction

set CUDA_VISIBLE_DEVICES=0

python finetune_constrained_completion_loss.py ^
  --mesh_file ..\data\allMeSH_2022\allMeSH_2022_random_5x.json ^
  --val_file ..\data\allMeSH_2022\allMeSH_2022_val_5000.json ^
  --max_samples 58174 ^
  --epochs 3 ^
  --batch_size 4 ^
  --learning_rate 2e-4 ^
  --use_4bit
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mesh_file` | required | Path to training JSON file |
| `--val_file` | optional | Path to separate validation JSON file |
| `--max_samples` | 500 | Maximum articles to use for training |
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 4 | Training batch size (reduce if OOM) |
| `--learning_rate` | 2e-4 | Learning rate for LoRA adapter |
| `--model_path` | `meta-llama/Llama-3.1-8B-Instruct` | Base model identifier |
| `--output_dir` | `./mesh_finetuned_constrained` | Directory to save adapter and vocabulary |
| `--use_4bit` | False | Enable QLoRA 4-bit quantization (recommended) |

### Training Presets

**Quick Test (100 articles, 1 epoch):**
```bash
python finetune_constrained_completion_loss.py ^
  --mesh_file ..\data\allMeSH_2022\allMeSH_2022_random_5x.json ^
  --val_file ..\data\allMeSH_2022\allMeSH_2022_val_5000.json ^
  --max_samples 100 --epochs 1 --batch_size 4 --use_4bit
```

**Balanced (1k articles, 2 epochs):**
```bash
python finetune_constrained_completion_loss.py ^
  --mesh_file ..\data\allMeSH_2022\allMeSH_2022_random_5x.json ^
  --val_file ..\data\allMeSH_2022\allMeSH_2022_val_5000.json ^
  --max_samples 1000 --epochs 2 --batch_size 4 --use_4bit
```

**Full Training (58k articles, 3 epochs):**
```bash
python finetune_constrained_completion_loss.py ^
  --mesh_file ..\data\allMeSH_2022\allMeSH_2022_random_5x.json ^
  --val_file ..\data\allMeSH_2022\allMeSH_2022_val_5000.json ^
  --max_samples 58174 --epochs 3 --batch_size 4 --use_4bit
```

### Training Output

After training, you will get:
```
./mesh_finetuned_constrained_YYYYMMDD_HHMMSS/
├── adapter_config.json           # LoRA adapter configuration
├── adapter_model.bin             # LoRA weights
├── mesh_vocabulary.json          # Extracted MeSH vocabulary (important!)
└── [other files...]
```

**Important:** Save the `mesh_vocabulary.json` path—you'll need it for extraction.

### Training Tips

- **OOM during training?** Reduce `--batch_size` to 2 or 1
- **Training slow?** Use `--epochs 1` for quick iteration
- **Model not improving?** Check that `--val_file` is separate from training

---

## Extraction

### Single-Temperature Extraction

**Script:** `mesh_extraction/extract_finetuned.py`

**Best for:** Quick extraction, deterministic output, lower latency.

```bash
cd BioXplorer-BioGPT

python mesh_extraction\extract_finetuned.py ^
  --mesh_file data\allMeSH_2022\allMeSH_2022_val_5000.json ^
  --base_model meta-llama/Llama-3.1-8B-Instruct ^
  --adapter_path mesh_extraction\mesh_finetuned_constrained_20260406_035727 ^
  --max_samples 5000 ^
  --output_file greedy_extraction.json
```
### Extraction Arguments

#### extract_finetuned.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--mesh_file` | required | Path to input JSON |
| `--base_model` | `meta-llama/Llama-3.1-8B-Instruct` | Base model identifier |
| `--adapter_path` | required | Path to fine-tuned adapter directory |
| `--max_samples` | 10 | Number of articles to extract |
| `--temperature` | 0.7 | Sampling temperature (0.0 = greedy) |
| `--max_tokens` | 1280 | Maximum new tokens to generate |

---

## Evaluation

### Evaluation Script

**File:** `mesh_extraction/evaluate.py`

**Purpose:** Compute precision, recall, F1, and other metrics.

### Mode 1: Evaluate Extraction Results File

```bash
cd BioXplorer-BioGPT

python mesh_extraction\evaluate.py ^
  --results_file greedy_extraction.json ^
  --output_file greedy_extraction_eval.json
```

### Mode 2: Evaluate Predictions Against Dataset Split

```bash
python mesh_extraction\evaluate.py ^
  --predictions_file my_predictions.json ^
  --split test ^
  --output_file test_eval.json
```

### Evaluation Output Example

```json
{
  "n_samples": 5000,
  "macro_avg": {
    "precision": 0.468,
    "recall": 0.534,
    "f1": 0.499
  },
  "micro_avg": {
    "precision": 0.575,
    "recall": 0.494,
    "f1": 0.531
  },
  "totals": {
    "true_positives": 1343,
    "false_positives": 1526,
    "false_negatives": 1171
  }
}
```

---

## Model Comparison

### Current Best (Constrained Model)

- Training: completion-only loss, vocabulary validation
- Result: 0.502 Macro F1, 0.499 Micro F1 on 200-sample validation
- Status: Production-ready
- Path: `mesh_extraction/mesh_finetuned_constrained_20260406_035727`

### Key Patterns

**Strengths:**
- Clean JSON output (0% invalid JSON)
- Good core concept detection
- Stable across different validation samples

**Weaknesses:**
- Misses exact MeSH granularity
- Weak on secondary/contextual concepts
- Weak on formal methodology labels

---

## Troubleshooting

### OOM (Out of Memory) During Training

**Solutions:**
1. Reduce batch size:
   ```bash
   --batch_size 2  # or 1
   ```
2. Reduce max samples:
   ```bash
   --max_samples 1000
   ```
3. Ensure `--use_4bit` is enabled

### Training Loss Not Decreasing

**Checks:**
1. Verify vocabulary is populated
2. Verify training data is being loaded
3. Check GPU is being used: `nvidia-smi`

### Extraction Very Slow

**Causes & Fixes:**
1. Using ensemble with multiple temperatures (normal, runs sequentially)
   - Expected: 3 temps × 8s per temp = ~24s per article
2. GPU not being used: `nvidia-smi`
3. Model not merged properly

### Invalid JSON Output

**Fixes:**
1. Model not trained properly (retrain with longer epochs)
2. Vocabulary not constrained during inference
3. Check extraction script's system prompt

---
