from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np
import evaluate
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# Bitext only has a "train" split — no test split.
# We carve it into train (80%) / eval (10%) / test (10%)
# ─────────────────────────────────────────────

raw = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"]

# Bitext columns: "instruction" (text), "intent" (label string)
# Rename "instruction" → "text" for consistency
raw = raw.rename_column("instruction", "text")

# Check what columns and intent labels exist
print("Columns:", raw.column_names)
print("Sample intents:", raw["intent"][:5])
print("Unique intents:", sorted(set(raw["intent"])))
print("Total samples:", len(raw))

# ─────────────────────────────────────────────
# 2. ENCODE STRING LABELS → INTEGER LABELS
# DeBERTa needs integer labels, not strings
# ─────────────────────────────────────────────

unique_intents = sorted(set(raw["intent"]))
label2id = {label: idx for idx, label in enumerate(unique_intents)}
id2label  = {idx: label for label, idx in label2id.items()}
num_labels = len(unique_intents)

print(f"\nNumber of intent classes: {num_labels}")
print("label2id:", label2id)

def encode_labels(batch):
    batch["label"] = [label2id[intent] for intent in batch["intent"]]
    return batch

raw = raw.map(encode_labels, batched=True)

# ─────────────────────────────────────────────
# 3. SPLIT INTO TRAIN / EVAL / TEST
# 80% train | 10% eval | 10% test
# ─────────────────────────────────────────────

split_1 = raw.train_test_split(test_size=0.2, seed=42)
split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)

train_ds = split_1["train"]     # 80%
eval_ds  = split_2["train"]     # 10%
test_ds  = split_2["test"]      # 10% — held out for final evaluation

print(f"\nTrain: {len(train_ds)} | Eval: {len(eval_ds)} | Test: {len(test_ds)}")

# ─────────────────────────────────────────────
# 4. LOAD MODEL + TOKENIZER
# ─────────────────────────────────────────────

model_name = "microsoft/deberta-v3-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,       # baked into model config — useful for inference later
    label2id=label2id,
)

# ─────────────────────────────────────────────
# 5. TOKENIZE
# ─────────────────────────────────────────────

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)
    # 128 is enough for short customer queries — faster than 256

train_ds = train_ds.map(tokenize, batched=True)
eval_ds  = eval_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ─────────────────────────────────────────────
# 6. METRICS
# Use macro F1 for multi-class (averages across all 27 intents equally)
# ─────────────────────────────────────────────

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(
            predictions=preds,
            references=labels
        )["accuracy"],
        "f1": f1_metric.compute(
            predictions=preds,
            references=labels,
            average="macro"     # macro = treat all 27 intents equally
        )["f1"],
    }

# ─────────────────────────────────────────────
# 7. TRAINING CONFIG
# ─────────────────────────────────────────────

args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=3e-6,
    warmup_steps=500,
    adam_epsilon=1e-5,
    max_grad_norm=1.0,
    weight_decay=0.01,
    optim="adamw_torch",
    tf32=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=20,
    fp16=False,
    bf16=False,
)

# ─────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ─────────────────────────────────────────────
# 9. FINAL EVALUATION ON HELD-OUT TEST SET
# ─────────────────────────────────────────────

print("\n── Final test set evaluation ──")
results = trainer.evaluate(test_ds)
print(results)

# ─────────────────────────────────────────────
# 10. SAVE MODEL
# ─────────────────────────────────────────────

trainer.save_model("./out/final_model")
tokenizer.save_pretrained("./out/final_model")
print("\nModel saved to ./out/final_model")

# ─────────────────────────────────────────────
# 11. QUICK INFERENCE TEST
# ─────────────────────────────────────────────

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./out/final_model",
    tokenizer="./out/final_model",
    top_k=3                 # show top 3 predicted intents
)

test_queries = [
    "I want to cancel my order",
    "Where is my delivery?",
    "I was charged twice for the same item",
    "How do I reset my password?",
    "I want a refund for my purchase",
]

print("\n── Sample predictions ──")
for query in test_queries:
    preds = classifier(query)[0]
    top   = preds[0]
    print(f"\nQuery:  {query}")
    print(f"Top intent: {top['label']} ({top['score']:.2%})")
    top3 = [(p["label"], f"{p['score']:.2%}") for p in preds]
    print(f"All top 3:  {top3}")

