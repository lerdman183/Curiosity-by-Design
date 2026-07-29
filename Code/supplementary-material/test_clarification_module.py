import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.environ["SCRATCH"], "Curiosity-by-Design", "hf_cache")
FINE_TUNED_DIR = os.path.join(os.environ["SCRATCH"], "Curiosity-by-Design", "llama3.3-70B-ft-clarification")
EVAL_SPLIT_PATH = os.path.join(BASE_DIR, "..", "..", "Datasets", "clarification_module_eval_split.json")
 
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
assert torch.cuda.is_available(), "CUDA is required for this script"

# 2. Load tokenizer + model + LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_DIR)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
    device_map="auto" if DEVICE.startswith("cuda") else None,
)
model = PeftModel.from_pretrained(base_model, FINE_TUNED_DIR)
