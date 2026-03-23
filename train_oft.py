#!/usr/bin/env python3

import os
import sys
import json
import time
import re
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import OFTConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm





AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_TO_ID = {v: k for k, v in AG_NEWS_LABELS.items()}



PROMPT_TEMPLATE = (
    "Classify the news into: World, Sports, Business, Sci/Tech. "
    "Output ONLY the category name, nothing else.\n\n"
    "News: Oil prices rose amid OPEC supply cuts.\nCategory: Business\n\n"
    "News: The team won the championship final.\nCategory: Sports\n\n"
    "News: {text}\nCategory: "
)




def compute_max_label_tokens(tokenizer) -> int:
    max_tokens = 0
    for label_name in AG_NEWS_LABELS.values():
        token_ids = tokenizer(label_name, add_special_tokens=False)["input_ids"]
        max_tokens = max(max_tokens, len(token_ids))

    return max_tokens + 1







def parse_args():
    parser = argparse.ArgumentParser(
        description="OFT Generative Finetuning on Qwen3-0.6B for AG News Classification"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Pretrained model name or path",
    )

    parser.add_argument(
        "--dataset_name", type=str, default="ag_news", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length for tokenization",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=1000,
        help="Number of training samples to use (0 = all)",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=100,
        help="Number of eval samples to use (0 = all)",
    )

    parser.add_argument("--oft_block_size", type=int, default=32, help="OFT block size")
    parser.add_argument(
        "--oft_r",
        type=int,
        default=0,
        help="OFT rank (number of blocks per layer, 0 = auto)",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="all-linear",
        help="Target modules for OFT adaptation",
    )
    parser.add_argument(
        "--coft", action="store_true", default=False, help="Use constrained OFT"
    )
    parser.add_argument(
        "--module_dropout", type=float, default=0.0, help="OFT module dropout"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
        help="Warmup ratio of total training steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=0,
        help="Max new tokens to generate (0 = auto from label token lengths)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--log_steps", type=int, default=50, help="Log every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help="Evaluate every N steps (0 = epoch end only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cuda, mps, cpu)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 mixed precision training",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use BF16 mixed precision training",
    )

    return parser.parse_args()







def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def get_dtype(args) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    elif args.fp16:
        return torch.float16
    return torch.float32


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)







class GenerativeClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_id = self.labels[idx]
        label_str = AG_NEWS_LABELS[label_id]


        label_ids = self.tokenizer(
            label_str + self.tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]



        prompt_shell = PROMPT_TEMPLATE.format(text="")
        shell_ids = self.tokenizer(
            prompt_shell, add_special_tokens=False
        )["input_ids"]


        text_budget = self.max_length - len(shell_ids) - len(label_ids)
        if text_budget < 10:
            text_budget = 10


        text_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=text_budget,
        )["input_ids"]


        truncated_text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
        prompt = PROMPT_TEMPLATE.format(text=truncated_text)
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]


        input_ids = prompt_ids + label_ids
        attention_mask = [1] * len(input_ids)


        if len(input_ids) > self.max_length:

            excess = len(input_ids) - self.max_length
            input_ids = input_ids[excess:]
            attention_mask = attention_mask[excess:]
            prompt_len = max(0, len(prompt_ids) - excess)
        else:
            prompt_len = len(prompt_ids)


        labels = list(input_ids)
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_id": label_id,
        }


def collate_fn(batch, pad_token_id):
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []
    label_ids = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len


        input_ids.append(
            torch.cat(
                [item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
        )
        attention_mask.append(
            torch.cat(
                [item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
        )
        labels.append(
            torch.cat(
                [item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]
            )
        )
        label_ids.append(item["label_id"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "label_ids": label_ids,
    }


def prepare_datasets(args, tokenizer, logger):
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    if args.train_samples > 0 and args.train_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.train_samples))
        logger.info(f"Subsampled training set to {args.train_samples} samples")
    if args.eval_samples > 0 and args.eval_samples < len(test_ds):
        test_ds = test_ds.shuffle(seed=args.seed).select(range(args.eval_samples))
        logger.info(f"Subsampled eval set to {args.eval_samples} samples")

    logger.info(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    train_dataset = GenerativeClassificationDataset(
        train_ds["text"], train_ds["label"], tokenizer, args.max_length
    )
    test_dataset = GenerativeClassificationDataset(
        test_ds["text"], test_ds["label"], tokenizer, args.max_length
    )

    return train_dataset, test_dataset







def parse_generated_label(generated_text: str) -> int:
    text = generated_text.strip().lower()


    for label_name, label_id in LABEL_TO_ID.items():
        if text.startswith(label_name.lower()):
            return label_id


    for label_name, label_id in LABEL_TO_ID.items():
        if label_name.lower() in text:
            return label_id


    partial_map = {
        "sci": 3,
        "tech": 3,
        "sport": 1,
        "business": 2,
        "busi": 2,
        "world": 0,
        "international": 0,
        "politic": 0,
    }
    for key, label_id in partial_map.items():
        if key in text:
            return label_id

    return -1


@torch.no_grad()
def evaluate_generative(model, tokenizer, dataset, device, args, logger=None, max_samples=None):
    model.eval()

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    eval_bs = args.eval_batch_size
    all_preds = []
    all_labels = []
    all_generated = []
    total_loss = 0.0
    loss_count = 0

    num_batches = (n + eval_bs - 1) // eval_bs

    for batch_idx in tqdm(range(num_batches), desc="Evaluating (generative)", leave=False):
        start = batch_idx * eval_bs
        end = min(start + eval_bs, n)
        batch_indices = list(range(start, end))
        cur_bs = len(batch_indices)


        batch_items = [dataset[i] for i in batch_indices]
        batch_label_ids = [item["label_id"] for item in batch_items]
        batch_texts = [dataset.texts[i] for i in batch_indices]


        pad_id = tokenizer.pad_token_id
        loss_batch = collate_fn(batch_items, pad_id)
        loss_input_ids = loss_batch["input_ids"].to(device)
        loss_attn_mask = loss_batch["attention_mask"].to(device)
        loss_labels = loss_batch["labels"].to(device)

        outputs = model(
            input_ids=loss_input_ids,
            attention_mask=loss_attn_mask,
            labels=loss_labels,
        )
        total_loss += outputs.loss.item() * cur_bs
        loss_count += cur_bs


        prompts = [PROMPT_TEMPLATE.format(text=t) for t in batch_texts]


        prompt_encodings = [
            tokenizer(
                p,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_length - args.max_new_tokens,
            )
            for p in prompts
        ]
        prompt_lengths = [len(enc["input_ids"]) for enc in prompt_encodings]
        max_prompt_len = max(prompt_lengths)


        gen_input_ids = []
        gen_attn_mask = []
        for enc in prompt_encodings:
            ids = enc["input_ids"]
            pad_len = max_prompt_len - len(ids)
            gen_input_ids.append([pad_id] * pad_len + ids)
            gen_attn_mask.append([0] * pad_len + [1] * len(ids))

        gen_input_ids = torch.tensor(gen_input_ids, dtype=torch.long, device=device)
        gen_attn_mask = torch.tensor(gen_attn_mask, dtype=torch.long, device=device)

        gen_outputs = model.generate(
            input_ids=gen_input_ids,
            attention_mask=gen_attn_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


        for j in range(cur_bs):
            gen_tokens = gen_outputs[j, max_prompt_len:]
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            pred_id = parse_generated_label(generated_text)

            all_preds.append(pred_id)
            all_labels.append(batch_label_ids[j])
            all_generated.append(generated_text)


            sample_idx = start + j + 1
            true_label = AG_NEWS_LABELS[batch_label_ids[j]]
            pred_label = AG_NEWS_LABELS.get(pred_id, f"UNKNOWN")
            mark = "✓" if pred_id == batch_label_ids[j] else "✗"
            print(
                f"  [{sample_idx:>{len(str(n))}}/{n}] {mark}  "
                f"True: {true_label:<10} | Generated: \"{generated_text:<20}\" | Pred: {pred_label}"
            )


        valid_so_far = [(p, l) for p, l in zip(all_preds, all_labels) if p != -1]
        if valid_so_far:
            running_acc = sum(1 for p, l in valid_so_far if p == l) / len(valid_so_far)
            print(f"  --- Running accuracy: {running_acc:.4f} ({sum(1 for p,l in valid_so_far if p==l)}/{len(valid_so_far)}) ---")


    valid_mask = [p != -1 for p in all_preds]
    valid_preds = [p for p, v in zip(all_preds, valid_mask) if v]
    valid_labels = [l for l, v in zip(all_labels, valid_mask) if v]
    unparseable_count = sum(1 for p in all_preds if p == -1)

    avg_loss = total_loss / max(loss_count, 1)

    if len(valid_preds) > 0:
        accuracy = accuracy_score(valid_labels, valid_preds)
        f1_macro = f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
        f1_weighted = f1_score(valid_labels, valid_preds, average="weighted", zero_division=0)
        precision = precision_score(valid_labels, valid_preds, average="macro", zero_division=0)
        recall = recall_score(valid_labels, valid_preds, average="macro", zero_division=0)
    else:
        accuracy = f1_macro = f1_weighted = precision = recall = 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
        "unparseable": unparseable_count,
        "total_evaluated": n,
        "valid_evaluated": len(valid_preds),
    }


    if len(valid_preds) > 0:
        target_names = [AG_NEWS_LABELS[i] for i in range(4)]
        report = classification_report(
            valid_labels, valid_preds, target_names=target_names, digits=4,
            zero_division=0, labels=list(range(4))
        )
        cm = confusion_matrix(valid_labels, valid_preds, labels=list(range(4)))
        metrics["classification_report"] = report
        metrics["confusion_matrix"] = cm.tolist()

    metrics["all_preds"] = all_preds
    metrics["all_labels"] = all_labels
    metrics["all_generated"] = all_generated

    model.train()
    return metrics







@torch.no_grad()
def qualitative_evaluation(model, tokenizer, device, args, logger=None):
    sample_texts = [

        "The United Nations Security Council held an emergency meeting to discuss the escalating tensions in the Middle East region.",
        "European leaders gathered in Brussels to negotiate a new trade agreement with Asian economies.",
        "Humanitarian aid organizations are rushing supplies to flood-affected areas in Southeast Asia.",

        "LeBron James scored 45 points in the NBA Finals, leading the Lakers to a decisive victory over the Celtics.",
        "The FIFA World Cup qualifiers saw several upsets as underdog teams defeated higher-ranked opponents.",
        "Tennis star Novak Djokovic won his 24th Grand Slam title at the Australian Open.",

        "Apple Inc. reported record quarterly revenue of $123 billion, driven by strong iPhone and services sales.",
        "The Federal Reserve announced another interest rate hike to combat persistent inflation in the economy.",

        "NASA's James Webb Space Telescope captured unprecedented images of distant galaxies formed shortly after the Big Bang.",
        "Researchers at MIT developed a new quantum computing algorithm that could revolutionize cryptography and drug discovery.",
    ]
    expected_labels = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]

    model.eval()
    results = []

    for text, expected in zip(sample_texts, expected_labels):
        prompt = PROMPT_TEMPLATE.format(text=text)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length - args.max_new_tokens,
        ).to(device)

        gen_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        gen_tokens = gen_ids[0, enc["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        pred_id = parse_generated_label(generated_text)

        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "expected": AG_NEWS_LABELS[expected],
            "generated": generated_text,
            "predicted": AG_NEWS_LABELS.get(pred_id, f"UNKNOWN({generated_text})"),
            "correct": pred_id == expected,
        }
        results.append(result)

    if logger:
        logger.info("\n" + "=" * 80)
        logger.info("QUALITATIVE EVALUATION — GENERATIVE RESULTS")
        logger.info("=" * 80)
        correct_count = sum(1 for r in results if r["correct"])
        logger.info(f"Correct: {correct_count}/{len(results)}")
        logger.info("-" * 80)
        for i, r in enumerate(results):
            status = "✓" if r["correct"] else "✗"
            logger.info(f"\n[{status}] Sample {i + 1}:")
            logger.info(f"  Text:       {r['text']}")
            logger.info(f"  Expected:   {r['expected']}")
            logger.info(f"  Generated:  \"{r['generated']}\"")
            logger.info(f"  Predicted:  {r['predicted']}")
        logger.info("=" * 80)

    return results







def plot_training_curves(train_losses, eval_metrics_history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "OFT Generative Finetuning — Qwen3-0.6B on AG News",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )


    ax = axes[0, 0]
    steps = [x["step"] for x in train_losses]
    losses = [x["loss"] for x in train_losses]
    ax.plot(steps, losses, color="#3498db", linewidth=1.0, alpha=0.5, label="Training Loss")

    if len(losses) > 10:
        window = min(len(losses) // 5, 50)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            smooth_steps = steps[window - 1 :]
            ax.plot(smooth_steps, smoothed, color="#e74c3c", linewidth=2.5, label="Smoothed")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)


    if eval_metrics_history:
        ax = axes[0, 1]
        eval_steps = [x["step"] for x in eval_metrics_history]
        eval_losses = [x["loss"] for x in eval_metrics_history]
        ax.plot(eval_steps, eval_losses, "o-", color="#2ecc71", linewidth=2, markersize=8)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Evaluation Loss", fontweight="bold")
        ax.grid(True, alpha=0.3)


    if eval_metrics_history:
        ax = axes[1, 0]
        accuracies = [x["accuracy"] for x in eval_metrics_history]
        f1_scores = [x["f1_macro"] for x in eval_metrics_history]

        ax.plot(eval_steps, accuracies, "s-", color="#9b59b6", linewidth=2, markersize=8, label="Accuracy")
        ax.plot(eval_steps, f1_scores, "D-", color="#e67e22", linewidth=2, markersize=8, label="F1 (Macro)")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Score")
        ax.set_title("Accuracy & F1 Score", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])


    ax = axes[1, 1]
    ax.text(
        0.5, 0.5,
        "See before_after_comparison.png\nfor detailed comparison",
        ha="center", va="center", fontsize=13, color="#7f8c8d",
        transform=ax.transAxes,
    )
    ax.set_title("Before vs After", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_before_after_comparison(before_metrics, after_metrics, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Before vs After OFT Generative Finetuning — Performance Comparison",
        fontsize=15, fontweight="bold", y=1.02,
    )


    ax = axes[0]
    names = ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "Precision", "Recall"]
    keys = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    bv = [before_metrics.get(k, 0) for k in keys]
    av = [after_metrics.get(k, 0) for k in keys]

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w / 2, bv, w, label="Before (Zero-shot)", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + w / 2, av, w, label="After (OFT Finetuned)", color="#2ecc71", alpha=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#c0392b")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, color="#27ae60")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison", fontweight="bold")
    ax.legend()
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis="y")


    ax = axes[1]
    if "confusion_matrix" in after_metrics:
        cm = np.array(after_metrics["confusion_matrix"])
        label_names = [AG_NEWS_LABELS[i] for i in range(4)]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=12, color="white" if cm[i, j] > thresh else "black")

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (After OFT)", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "before_after_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_qualitative_comparison(before_results, after_results, output_dir):
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.axis("off")
    ax.set_title(
        "Qualitative — Before vs After OFT Generative Finetuning",
        fontsize=15, fontweight="bold", pad=20,
    )

    col_labels = ["#", "Text (truncated)", "True", "Before (generated)", "After (generated)"]
    rows = []
    for i, (b, a) in enumerate(zip(before_results, after_results)):
        short_text = b["text"][:55] + "..." if len(b["text"]) > 55 else b["text"]
        bm = "✓" if b["correct"] else "✗"
        am = "✓" if a["correct"] else "✗"
        rows.append([
            str(i + 1),
            short_text,
            b["expected"],
            f'{bm} "{b["generated"]}"',
            f'{am} "{a["generated"]}"',
        ])

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        if before_results[i]["correct"]:
            table[i + 1, 3].set_facecolor("#d5f5e3")
        else:
            table[i + 1, 3].set_facecolor("#fadbd8")
        if after_results[i]["correct"]:
            table[i + 1, 4].set_facecolor("#d5f5e3")
        else:
            table[i + 1, 4].set_facecolor("#fadbd8")

    col_widths = [0.03, 0.35, 0.07, 0.22, 0.22]
    for j, cw in enumerate(col_widths):
        for i in range(len(rows) + 1):
            table[i, j].set_width(cw)

    path = os.path.join(output_dir, "qualitative_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path







def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    logger.info("=" * 70)
    logger.info("  OFT Generative Finetuning — Qwen3-0.6B on AG News")
    logger.info("  (Causal LM — model generates classification labels)")
    logger.info("=" * 70)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    set_seed(args.seed)
    device = get_device(args.device)
    compute_dtype = get_dtype(args)
    logger.info(f"Device: {device} | Dtype: {compute_dtype}")


    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token = eos_token: '{tokenizer.pad_token}'")


    if args.max_new_tokens <= 0:
        args.max_new_tokens = compute_max_label_tokens(tokenizer)
        logger.info(f"Auto max_new_tokens = {args.max_new_tokens} (from label token lengths)")
    else:
        logger.info(f"Manual max_new_tokens = {args.max_new_tokens}")


    for label_name in AG_NEWS_LABELS.values():
        tids = tokenizer(label_name, add_special_tokens=False)["input_ids"]
        logger.info(f"  Label '{label_name}' -> {len(tids)} tokens: {tids}")


    logger.info(f"Loading base model (CausalLM): {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info("Base CausalLM model loaded.")


    train_dataset, test_dataset = prepare_datasets(args, tokenizer, logger)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        num_workers=0,
        pin_memory=True,
    )





    logger.info("\n" + "=" * 60)
    logger.info("  STEP 1: Evaluating BASE model (ZERO-SHOT, before finetuning)")
    logger.info("=" * 60)

    model.to(device)

    logger.info("Running zero-shot generative evaluation...")
    before_metrics = evaluate_generative(
        model, tokenizer, test_dataset, device, args, logger,
        max_samples=min(500, len(test_dataset)),
    )
    logger.info(f"Zero-shot Results (on {before_metrics['total_evaluated']} samples):")
    logger.info(f"  Accuracy:       {before_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (Macro):     {before_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (Weighted):  {before_metrics['f1_weighted']:.4f}")
    logger.info(f"  Precision:      {before_metrics['precision_macro']:.4f}")
    logger.info(f"  Recall:         {before_metrics['recall_macro']:.4f}")
    logger.info(f"  Loss:           {before_metrics['loss']:.4f}")
    logger.info(f"  Unparseable:    {before_metrics['unparseable']}/{before_metrics['total_evaluated']}")
    if "classification_report" in before_metrics:
        logger.info(f"\n{before_metrics['classification_report']}")

    logger.info("\nRunning qualitative evaluation (base model, zero-shot)...")
    before_qualitative = qualitative_evaluation(model, tokenizer, device, args, logger)

    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()




    logger.info("\n" + "=" * 60)
    logger.info("  STEP 2: Applying OFT Adapter and Training (Generative)")
    logger.info("=" * 60)

    oft_config = OFTConfig(
        r=args.oft_r if args.oft_r > 0 else 0,
        oft_block_size=args.oft_block_size,
        target_modules=args.target_modules,
        module_dropout=args.module_dropout,
        coft=args.coft,
        bias="none",
        task_type=TaskType.CAUSAL_LM,

    )

    logger.info(f"OFT Config:\n{oft_config}")

    model = get_peft_model(model, oft_config)

    trainable_params, total_params = count_parameters(model)
    logger.info(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.4f}%)"
    )
    model.print_trainable_parameters()
    model.to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = (len(train_loader) * args.num_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info(f"Total steps: {total_steps} | Warmup: {warmup_steps} | Steps/epoch: {len(train_loader)}")


    use_amp = args.fp16 or args.bf16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    amp_dtype = compute_dtype if use_amp else torch.float32


    train_losses = []
    eval_metrics_history = []
    global_step = 0
    best_acc = 0.0
    start_time = time.time()

    logger.info("\n--- Training started ---")
    model.train()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=True)
        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp and device.type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            epoch_steps += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                cur_loss = loss.item() * args.gradient_accumulation_steps
                train_losses.append({"step": global_step, "loss": cur_loss, "epoch": epoch + 1})

                progress.set_postfix(
                    loss=f"{cur_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                if global_step % args.log_steps == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Step {global_step}/{total_steps} | Epoch {epoch + 1} | "
                        f"Loss: {cur_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Time: {elapsed:.0f}s"
                    )


                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_m = evaluate_generative(
                        model, tokenizer, test_dataset, device, args, logger,
                        max_samples=min(500, len(test_dataset)),
                    )
                    eval_m["step"] = global_step
                    eval_m["epoch"] = epoch + 1
                    eval_metrics_history.append(eval_m)
                    logger.info(
                        f"  [Eval@{global_step}] Loss: {eval_m['loss']:.4f} | "
                        f"Acc: {eval_m['accuracy']:.4f} | F1: {eval_m['f1_macro']:.4f} | "
                        f"Unparseable: {eval_m['unparseable']}"
                    )
                    if eval_m["accuracy"] > best_acc:
                        best_acc = eval_m["accuracy"]
                        model.save_pretrained(os.path.join(args.output_dir, "best_model"))
                        tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
                        logger.info(f"  ★ New best accuracy: {best_acc:.4f}")


        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(f"\n--- Epoch {epoch + 1} done | Avg Loss: {avg_epoch_loss:.4f} ---")

        eval_m = evaluate_generative(
            model, tokenizer, test_dataset, device, args, logger,
            max_samples=min(500, len(test_dataset)),
        )
        eval_m["step"] = global_step
        eval_m["epoch"] = epoch + 1
        eval_metrics_history.append(eval_m)

        logger.info(
            f"  [Eval@epoch{epoch + 1}] Loss: {eval_m['loss']:.4f} | "
            f"Acc: {eval_m['accuracy']:.4f} | F1: {eval_m['f1_macro']:.4f} | "
            f"Unparseable: {eval_m['unparseable']}"
        )

        if eval_m["accuracy"] > best_acc:
            best_acc = eval_m["accuracy"]
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            logger.info(f"  ★ New best accuracy: {best_acc:.4f}")

    total_time = time.time() - start_time
    logger.info(f"\n--- Training completed in {total_time:.1f}s ---")




    logger.info("\n" + "=" * 60)
    logger.info("  STEP 3: Final Evaluation (AFTER OFT finetuning)")
    logger.info("=" * 60)

    after_metrics = evaluate_generative(
        model, tokenizer, test_dataset, device, args, logger,
        max_samples=len(test_dataset),
    )
    logger.info(f"Finetuned Model Results (on {after_metrics['total_evaluated']} samples):")
    logger.info(f"  Accuracy:       {after_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (Macro):     {after_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (Weighted):  {after_metrics['f1_weighted']:.4f}")
    logger.info(f"  Precision:      {after_metrics['precision_macro']:.4f}")
    logger.info(f"  Recall:         {after_metrics['recall_macro']:.4f}")
    logger.info(f"  Loss:           {after_metrics['loss']:.4f}")
    logger.info(f"  Unparseable:    {after_metrics['unparseable']}/{after_metrics['total_evaluated']}")

    if "classification_report" in after_metrics:
        logger.info(f"\n{after_metrics['classification_report']}")

    logger.info("\nRunning qualitative evaluation (finetuned)...")
    after_qualitative = qualitative_evaluation(model, tokenizer, device, args, logger)




    logger.info("\n" + "=" * 60)
    logger.info("  STEP 4: Generating Plots & Summary")
    logger.info("=" * 60)

    p = plot_training_curves(train_losses, eval_metrics_history, args.output_dir)
    logger.info(f"Training curves: {p}")

    p = plot_before_after_comparison(before_metrics, after_metrics, args.output_dir)
    logger.info(f"Before/After comparison: {p}")

    p = plot_qualitative_comparison(before_qualitative, after_qualitative, args.output_dir)
    logger.info(f"Qualitative comparison: {p}")


    def _clean_metrics(m):
        return {k: v for k, v in m.items()
                if k not in ("classification_report", "confusion_matrix",
                             "all_preds", "all_labels", "all_generated")}

    summary = {
        "model": args.model_name,
        "dataset": args.dataset_name,
        "method": "OFT (Orthogonal Finetuning) — Generative / Causal LM",
        "approach": "Model generates classification labels as text (no classification head)",
        "prompt_template": PROMPT_TEMPLATE,
        "labels": AG_NEWS_LABELS,
        "oft_config": {
            "block_size": args.oft_block_size,
            "r": args.oft_r,
            "target_modules": args.target_modules,
            "coft": args.coft,
            "module_dropout": args.module_dropout,
            "task_type": "CAUSAL_LM",
        },
        "training_config": {
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
        },
        "parameters": {
            "trainable": trainable_params,
            "total": total_params,
            "trainable_pct": 100 * trainable_params / total_params,
        },
        "before_finetuning": _clean_metrics(before_metrics),
        "after_finetuning": _clean_metrics(after_metrics),
        "improvement": {
            "accuracy": after_metrics["accuracy"] - before_metrics["accuracy"],
            "f1_macro": after_metrics["f1_macro"] - before_metrics["f1_macro"],
        },
        "best_accuracy": best_acc,
        "training_time_s": total_time,
        "train_loss_history": train_losses,
        "eval_history": [_clean_metrics(m) for m in eval_metrics_history],
        "qualitative": {
            "before": before_qualitative,
            "after": after_qualitative,
        },
    }

    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to: {summary_path}")


    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved to: {final_dir}")


    logger.info("\n" + "=" * 70)
    logger.info("  FINAL COMPARISON — OFT GENERATIVE FINETUNING")
    logger.info("=" * 70)
    logger.info(f"  {'Metric':<25} {'Before':>12} {'After':>12} {'Δ':>12}")
    logger.info(f"  {'-' * 61}")
    for metric in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro", "loss"]:
        bv = before_metrics.get(metric, 0)
        av = after_metrics.get(metric, 0)
        delta = av - bv
        sign = "+" if delta > 0 else ""
        logger.info(f"  {metric:<25} {bv:>12.4f} {av:>12.4f} {sign}{delta:>11.4f}")

    logger.info(f"\n  Unparseable (before): {before_metrics['unparseable']}/{before_metrics['total_evaluated']}")
    logger.info(f"  Unparseable (after):  {after_metrics['unparseable']}/{after_metrics['total_evaluated']}")
    logger.info(f"  Trainable Params: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
    logger.info(f"  Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Best Accuracy: {best_acc:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
