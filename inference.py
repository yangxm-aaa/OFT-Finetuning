#!/usr/bin/env python3

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_TO_ID = {v: k for k, v in AG_NEWS_LABELS.items()}

PROMPT_TEMPLATE = (
    "Classify the following news article into one of these categories: "
    "World, Sports, Business, Sci/Tech.\n\n"
    "Article: {text}\n\n"
    "Category: "
)


def parse_generated_label(text: str) -> tuple:
    text = text.strip().lower()
    for name, lid in LABEL_TO_ID.items():
        if text.startswith(name.lower()):
            return name, lid
    for name, lid in LABEL_TO_ID.items():
        if name.lower() in text:
            return name, lid
    partial = {"sci": 3, "tech": 3, "sport": 1, "business": 2, "world": 0}
    for key, lid in partial.items():
        if key in text:
            return AG_NEWS_LABELS[lid], lid
    return f"Unknown({text})", -1


def load_model(model_path, device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading model from: {model_path}")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True, torch_dtype=torch.float16
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}!")
    return model, tokenizer, device


@torch.no_grad()
def predict(model, tokenizer, text, device, max_new_tokens=5):
    prompt = PROMPT_TEMPLATE.format(text=text)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                    truncation=True, max_length=250).to(device)

    gen_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = gen_ids[0, enc["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    label_name, label_id = parse_generated_label(generated_text)

    return {
        "generated_text": generated_text,
        "label": label_name,
        "label_id": label_id,
    }


def main():
    parser = argparse.ArgumentParser(description="OFT Generative Classification Inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path, args.device)

    if args.text:
        result = predict(model, tokenizer, args.text, device, args.max_new_tokens)
        print(f"\nInput:     {args.text}")
        print(f"Generated: \"{result['generated_text']}\"")
        print(f"Label:     {result['label']}")

    elif args.interactive:
        print("\n--- Interactive Generative Classification ---")
        print("Type text and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input(">>> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predict(model, tokenizer, text, device, args.max_new_tokens)
            print(f"  Generated: \"{result['generated_text']}\"")
            print(f"  Label:     {result['label']}\n")
    else:
        print("Please specify --text or --interactive")


if __name__ == "__main__":
    main()
