import json
import torch
import os
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.environ["SCRATCH"], "Curiosity-by-Design", "hf_cache")
FINE_TUNED_DIR = os.path.join(os.environ["SCRATCH"], "Curiosity-by-Design", "llama3.3-70B-ft-clarification-gerrit")
EVAL_SPLIT_PATH = os.path.join(BASE_DIR, "..", "..", "Datasets", "clarification_module_eval_split.json")
 
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

assert torch.cuda.is_available(), "CUDA is required for this script"

# 2. Load evaluation examples + tokenizer + base and ft model
def load_eval_examples():
    #Load the exact 30% slice saved by clarification_module.py
    with open(EVAL_SPLIT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_tokenizer():
    # Loaded from FINE_TUNED_DIR, not the base checkpoint
    tok = AutoTokenizer.from_pretrained(FINE_TUNED_DIR)
    tok.pad_token = tok.eos_token
    return tok

def bnb_config():
    # Must use 4-bit quantization to load Llama3.3-70B onto one h100
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_base_model():
    # Load the untrained model for comparison against the trained model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_config(),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    )
    model.eval()
    return model

def load_finetuned_model():
    # Base checkpoint + the trained LoRA adapter
    base = load_base_model()
    model = PeftModel.from_pretrained(base, FINE_TUNED_DIR)
    model.eval()
    return model


# 3. Generate responses and evaluate them
def generate_response(model, tokenizer, prompt, max_new_tokens=64):
    # Chat-template formatting, matching the updated TextDataset: the
    # model was trained on this structure, so eval has to use it too
    messages = [{"role": "user", "content": prompt.strip()}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("Model produced NaN or Inf in logits before generation.")
 
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=100,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
 
    generated = out[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def is_clarifying_question(response_text, source_prompt):
    """
    A clarifying question if ends in '?' AND shares a token (variable/function/keyword name) with
    the source prompt, so generic non-specific questions don't count.
    """
    text = response_text.strip()
    if not text.endswith("?"):
        return False

    # pattern: letter or underscore followed by 2 or more letters, numbers, or underscores
    prompt_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", source_prompt))
    question_tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text))
    # If 1 or more tokens are shared in both sets, this will return true
    return len(prompt_tokens & question_tokens) > 0

def evaluate(model, tokenizer, eval_examples, label):
    question_count = 0
    results = []
    for ex in eval_examples:
        # Generate response and check if it is a clarifying question
        response = generate_response(model, tokenizer, ex["input"])
        is_question = is_clarifying_question(response, ex["input"])
        question_count += int(is_question)

        # Print input + output + if it is a question
        print("─" * 40)
        print(f"Prompt:   {ex['input']}")
        print(f"Response: {response}")
        print(f"Is clarifying question: {is_question}\n")

        # Add the results to the list
        results.append({
            "prompt": ex["input"],
            "response": response,
            "is_clarifying_question": is_question,
        })

    # Record the length of examples used, the percentage of clarifying questions generated
    total = len(eval_examples)
    pct = 100 * question_count / total if total else 0.0
    # Print that result and return
    print(f"[{label}] {question_count}/{total} responses were clarifying questions ({pct:.1f}%)")
    return question_count, total, results

if __name__ == "__main__":
    eval_examples = load_eval_examples()
    print(f"Loaded {len(eval_examples)} evaluation examples from {EVAL_SPLIT_PATH}")

    tokenizer = load_tokenizer()

    # Generate and evaluate responses from baseline model
    print("\nLoading untrained (base) model...")
    base_model = load_base_model()
    base_count, base_total, base_results = evaluate(base_model, tokenizer, eval_examples, "Untrained baseline")
    del base_model
    torch.cuda.empty_cache()

    # Generate and evaluate responses from fine tuned model
    print("\nLoading untrained (base) model...")
    finetuned_model = load_finetuned_model()
    ft_count, ft_total, ft_results = evaluate(base_model, tokenizer, eval_examples, "Untrained baseline")
    del finetuned_model
    torch.cuda.empty_cache()

    print("\n--- Summary ---")
    print(f"Baseline:   {base_count}/{base_total} ({100 * base_count / base_total:.2f}%)")
    print(f"Fine-tuned: {ft_count}/{ft_total} ({100 * ft_count / ft_total:.2f}%)")

    with open(os.path.join(BASE_DIR, "clarification_module_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "baseline": base_results,
            "finetuned": ft_results,
            "summary": {
                "baseline_question_count": base_count,
                "finetuned_question_count": ft_count,
                "total": ft_total,
            },
        }, f, ensure_ascii=False, indent=2)
 
    print(f"\nSaved full results to clarification_module_eval_results.json")