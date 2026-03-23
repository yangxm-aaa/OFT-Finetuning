# OFT Finetuning — Qwen3-0.6B on AG News Classification

This project demonstrates **Orthogonal Finetuning (OFT)** on the **Qwen3-0.6B** language model for **text classification** on the **AG News** dataset (4 classes: World, Sports, Business, Sci/Tech).

The model is finetuned in a **generative** manner — it learns to generate the classification label as text (e.g. "Sports"), rather than using a classification head.

## What is OFT?

**Orthogonal Finetuning (OFT)** is a parameter-efficient finetuning (PEFT) method that learns orthogonal transformations applied multiplicatively to pretrained weight matrices. Unlike LoRA which uses additive low-rank updates, OFT preserves the **hyperspherical energy** of neurons, leading to better knowledge preservation and reduced catastrophic forgetting.

Key advantages:
- **Preserves pretraining knowledge** via orthogonal structure constraint
- **Parameter efficient** — only a small fraction of parameters are trained
- **Better generalization** thanks to structural inductive bias

## Project Structure

```
oft_finetune/
├── train_oft.py               # Main training script (complete pipeline)
├── inference.py               # Inference script for the finetuned model
├── plot_confusion_matrix.py   # Plot the confusion matrix
├── plot_training_curves.py    # Plot the training curve
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── output/                    # Generated after training
    ├── training_curves.png
    ├── before_after_comparison.png
    ├── qualitative_comparison.png
    ├── results_summary.json
    ├── training.log
    ├── best_model/
    └── final_model/
```

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Training

### Quick Start (Default Settings)
```bash
python train_oft.py
```

### Full Training (All Data)
```bash
python train_oft.py \
  --train_samples 0 \
  --eval_samples 0 \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 5e-5 \
  --bf16
```

### GPU with Mixed Precision
```bash
python train_oft.py \
  --device cuda \
  --bf16 \
  --batch_size 8 \
  --gradient_accumulation_steps 4
```

### Custom OFT Configuration
```bash
python train_oft.py \
  --oft_block_size 16 \
  --coft \
  --module_dropout 0.1 \
  --target_modules "all-linear"
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-0.6B` | Pretrained model |
| `--oft_block_size` | `32` | OFT block size |
| `--target_modules` | `all-linear` | Modules to apply OFT |
| `--coft` | `False` | Use constrained OFT |
| `--num_epochs` | `3` | Training epochs |
| `--batch_size` | `4` | Training batch size |
| `--eval_batch_size` | `4` | Evaluation batch size |
| `--learning_rate` | `5e-5` | Learning rate |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation |
| `--train_samples` | `1000` | Training samples (0=all) |
| `--eval_samples` | `100` | Eval samples (0=all) |
| `--bf16` | `False` | BF16 mixed precision |
| `--fp16` | `False` | FP16 mixed precision |

## Inference

```bash
# Single prediction
python inference.py --model_path ./output/best_model --text "Apple stock surged 5% after earnings beat"

# Interactive mode
python inference.py --model_path ./output/best_model --interactive
```

## What the Training Script Does

1. **Loads Qwen3-0.6B** as a CausalLM base model (no classification head)
2. **Evaluates the base model** zero-shot on AG News test set (before finetuning)
3. **Applies OFT adapter** via HuggingFace PEFT library (`OFTConfig` + `TaskType.CAUSAL_LM`)
4. **Trains** the model to generate label text (e.g. "Sports") with prompt masking (loss only on label tokens)
5. **Evaluates the finetuned model** by generating labels and parsing them (after finetuning)
6. **Generates comparison plots**:
   - Training loss curve (raw + smoothed)
   - Evaluation loss over training
   - Accuracy & F1 score progression
   - Before vs After bar chart comparison
   - Confusion matrix
   - Qualitative results table (showing generated text before & after)
7. **Saves everything** to the output directory

## Output Files

After training, the `output/` directory will contain:

| File | Description |
|------|-------------|
| `training_curves.png` | Training loss, eval loss, accuracy & F1 |
| `before_after_comparison.png` | Bar chart + confusion matrix |
| `qualitative_comparison.png` | Sample predictions table |
| `results_summary.json` | All metrics, configs, and history |
| `training.log` | Full training log |
| `best_model/` | Best checkpoint (by accuracy) |
| `final_model/` | Final checkpoint |

## References

- [OFT Paper: Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280)
- [BOFT Paper: Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization](https://arxiv.org/abs/2311.06243)
- [HuggingFace PEFT Library](https://github.com/huggingface/peft)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)
