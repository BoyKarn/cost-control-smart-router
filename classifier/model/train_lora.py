"""
LoRA training script for the Router Classifier Project.

This script replaces the original full fine-tuning approach to reduce GPU memory 
usage on kaggle.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import set_seed

# =============================
# 1. BASIC SETUP
# =============================

set_seed(42)

MODEL_NAME = "google/gemma-2b"
OUTPUT_DIR = "./results"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =============================
# 2. LOAD DATASET
# =============================

dataset = load_dataset("akorn14/router-classifier-dataset")

print(dataset)

# Label mappings
LABEL2ID = {
    "simple": 0,
    "medium": 1,
    "complex": 2,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def encode_labels(example):
    label = example["label"]

    # Merge the fourth class into "complex"
    if label == "ultra_complex":
        label = "complex"

    example["label"] = LABEL2ID[label]
    return example


# Convert string labels to integers
dataset = dataset.map(encode_labels)

# (Optional) Reduce dataset size while debugging
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(4000))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(500))
# =============================
# 3. TOKENIZER
# =============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    return tokenizer(
        examples["prompt"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.rename_column("label","labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# =============================
# 4. LOAD MODEL (SAFE MODE)
# =============================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    token=hf_token
)

# IMPORTANT FIXES
model.config.use_cache = False  # required for gradient checkpointing

# =============================
# 5. LoRA CONFIGURATION
# =============================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =============================
# 6. TRAINING ARGUMENTS
# =============================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,

    num_train_epochs=2,

    learning_rate=5e-6,  # stable for LoRA
    warmup_ratio=0.05,

    fp16=True,
    bf16=False,

    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    gradient_checkpointing=True,
    report_to="none",

    load_best_model_at_end=True,
)

# =============================
# 7. TRAINER
# =============================

data_collator = DataCollatorWithPadding(
    tokenizer = tokenizer
)

import evaluate

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return accuracy.compute(
        predictions=predictions,
        references=labels,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# =============================
# 8. TRAIN
# =============================

print("\n🚀 Starting training...\n")
print("=" * 60)
print("Starting LoRA training...")
print(f"Training examples: {len(dataset['train'])}")
print(f"Validation examples: {len(dataset['validation'])}")
print("=" * 60)
trainer.train()

# =============================
# 9. SAVE MODEL
# =============================

print("\n💾 Saving model...")
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

# =============================
# 10. QUICK INFERENCE TEST
# =============================

def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    return {
        "prediction": ID2LABEL[prediction],
        "probabilities": probs.cpu().numpy(),
    }
print("\n🔍 Inference Test:")
print(predict("This is a test message"))