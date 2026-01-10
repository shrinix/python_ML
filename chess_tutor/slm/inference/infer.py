import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tutor.slm.slm_config import (
    OUTPUT_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE_PRIMARY,
    TEMPERATURE_RETRY,
    MAX_RETRIES
)
from chess_tutor.slm.model.prompt import format_prompt
from .validator import is_valid_output, parse_json


tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    torch_dtype=torch.float32,
    device_map="cpu"
)
# Robustly ensure pad_token and eos_token are set (fix for IndexError on MPS)
if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
if tokenizer.eos_token is None:
    tokenizer.eos_token = tokenizer.pad_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
print("pad_token:", tokenizer.pad_token, "id:", tokenizer.pad_token_id)
print("eos_token:", tokenizer.eos_token, "id:", tokenizer.eos_token_id)

def tutor_response(input_payload):
    prompt = format_prompt(input_payload)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for attempt in range(MAX_RETRIES + 1):

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE_PRIMARY if attempt == 0 else TEMPERATURE_RETRY,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        if is_valid_output(text):
            return parse_json(text)

    return {
        "explanation": "Let’s focus on the principle behind the decision rather than specific moves.",
        "reflective_question": "Which principle should guide your choice here?",
        "key_takeaway": "Strong chess decisions come from principles, not memorized lines."
    }


import os

if __name__ == "__main__":
    # Example: load a test payload from the new slm/data/tutor_train.jsonl
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    TEST_FILE = os.path.join(DATA_DIR, "tutor_train.jsonl")
    if os.path.exists(TEST_FILE):
        with open(TEST_FILE) as f:
            for line in f:
                sample = None
                try:
                    sample = __import__('json').loads(line)
                except Exception:
                    continue
                if sample and "input" in sample:
                    payload = sample["input"]
                    break
            else:
                payload = None
    else:
        payload = None

    if payload:
        print(tutor_response(payload))
    else:
        print("No valid test payload found in tutor_train.jsonl.")