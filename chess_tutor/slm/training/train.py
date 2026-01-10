import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def tokenize(example):
    prompt = format_prompt(example["input"])
    completion = str(example["output"])
    text = prompt + completion
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_SEQ_LEN)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

import os
from chess_tutor.slm.slm_config import (
    MODEL_NAME, OUTPUT_DIR, MAX_SEQ_LEN, TRAIN_NUM_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_GRAD_ACCUM, TRAIN_LEARNING_RATE, TRAIN_SAMPLE_LIMIT
)
from chess_tutor.slm.model.prompt import format_prompt

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if torch.backends.mps.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="mps"
    )
elif torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Use the new data path under slm/data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_FILE = os.path.join(DATA_DIR, "tutor_train.jsonl")
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
dataset = load_dataset("json", data_files={"train": TRAIN_FILE})
dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# Optionally limit number of training samples for fast test
if TRAIN_SAMPLE_LIMIT is not None:
    dataset["train"] = dataset["train"].select(range(min(TRAIN_SAMPLE_LIMIT, len(dataset["train"]))))

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=TRAIN_GRAD_ACCUM,
    num_train_epochs=TRAIN_NUM_EPOCHS,
    learning_rate=TRAIN_LEARNING_RATE,
    # fp16 is not supported on MPS, so we disable it
    logging_steps=20,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)