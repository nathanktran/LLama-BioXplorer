import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("torch:", torch.__version__)
print("transformers:", __import__("transformers").__version__)
print("cuda available:", torch.cuda.is_available())

TRAIN_PATH = Path("data/allMeSH_2022/allMeSH_2022_random_5x.json")
VAL_PATH = Path("data/allMeSH_2022/allMeSH_2022_val_5000.json")
TEST_PATH = Path("data/allMeSH_2022/allMeSH_2022_test_5000.json")

for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset: {p}")


def load_articles(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("articles", [])

train_articles = load_articles(TRAIN_PATH)
val_articles = load_articles(VAL_PATH)
test_articles = load_articles(TEST_PATH)

print("train:", len(train_articles))
print("val  :", len(val_articles))
print("test :", len(test_articles))

def to_frame(articles):
    rows = []
    for a in articles:
        rows.append(
            {
                "pmid": a.get("pmid", ""),
                "title": a.get("title", ""),
                "abstractText": a.get("abstractText", ""),
                "meshMajor": a.get("meshMajor", []),
                "journal": a.get("journal", ""),
                "year": a.get("year", ""),
            }
        )
    return pd.DataFrame(rows)


def duplicate_count_safe(df: pd.DataFrame) -> int:
    dedup_df = df.copy()
    dedup_df["meshMajor"] = dedup_df["meshMajor"].apply(
        lambda v: tuple(v) if isinstance(v, list) else v
    )
    return int(dedup_df.duplicated().sum())


train_df = to_frame(train_articles)
val_df = to_frame(val_articles)
test_df = to_frame(test_articles)

for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{name} shape: {df.shape}")
    print("columns:", list(df.columns))
    print("null abstracts:", df["abstractText"].isna().sum())
    print("duplicate rows:", duplicate_count_safe(df))

train_df.head(2)

def clean_split(df: pd.DataFrame):
    texts = []
    labels = []
    for abstract, mesh in zip(df["abstractText"], df["meshMajor"]):
        text = "" if pd.isna(abstract) else str(abstract).strip()
        mesh_list = mesh if isinstance(mesh, list) else []
        if text:
            texts.append(text)
            labels.append(mesh_list)
    return texts, labels

train_texts, train_labels_raw = clean_split(train_df)
val_texts, val_labels_raw = clean_split(val_df)
test_texts, test_labels_raw = clean_split(test_df)

print("clean train:", len(train_texts))
print("clean val  :", len(val_texts))
print("clean test :", len(test_texts))
print("example labels:", train_labels_raw[0][:10])

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_labels_raw).astype("float32")
known_terms = set(mlb.classes_)

def keep_known_terms(label_lists, known):
    return [[label for label in labels if label in known] for labels in label_lists]

val_labels = mlb.transform(keep_known_terms(val_labels_raw, known_terms)).astype("float32")
test_labels = mlb.transform(keep_known_terms(test_labels_raw, known_terms)).astype("float32")

print("num classes:", len(mlb.classes_))
print("train labels:", train_labels.shape)
print("val labels  :", val_labels.shape)
print("test labels :", test_labels.shape)

checkpoint = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(mlb.classes_),
    problem_type="multi_label_classification",
)

print("checkpoint:", checkpoint)
print("num labels:", model.config.num_labels)

class MeSHDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": label,
        }

MAX_LEN = 256

train_dataset = MeSHDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
val_dataset = MeSHDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)
test_dataset = MeSHDataset(test_texts, test_labels, tokenizer, max_len=MAX_LEN)

print("dataset sizes:", len(train_dataset), len(val_dataset), len(test_dataset))

THRESHOLD = 0.25

def multi_label_metrics(logits, labels, threshold=THRESHOLD):
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    y_pred = (probs >= threshold).astype(int)
    y_true = labels.astype(int)

    metrics = {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }

    # ROC-AUC is undefined for label columns that contain only one class.
    valid_auc_cols = [
        i for i in range(y_true.shape[1])
        if np.unique(y_true[:, i]).shape[0] > 1
    ]
    if valid_auc_cols:
        metrics["roc_auc_macro"] = roc_auc_score(
            y_true[:, valid_auc_cols],
            probs[:, valid_auc_cols],
            average="macro",
        )
    else:
        metrics["roc_auc_macro"] = float("nan")
    metrics["roc_auc_valid_labels"] = len(valid_auc_cols)

    return metrics


def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    return multi_label_metrics(logits, eval_pred.label_ids, threshold=THRESHOLD)

import inspect

# Build TrainingArguments kwargs in a version-compatible way.
common_args = {
    "output_dir": "biomedbert_mesh_outputs",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "logging_strategy": "steps",
    "logging_steps": 100,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "greater_is_better": True,
    "fp16": torch.cuda.is_available(),
    "report_to": "none",
}

training_sig_params = inspect.signature(TrainingArguments.__init__).parameters
if "evaluation_strategy" in training_sig_params:
    common_args["evaluation_strategy"] = "epoch"
elif "eval_strategy" in training_sig_params:
    common_args["eval_strategy"] = "epoch"
else:
    raise RuntimeError("Neither 'evaluation_strategy' nor 'eval_strategy' is supported by this transformers version")

training_args = TrainingArguments(**common_args)

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset,
    "compute_metrics": compute_metrics,
}

trainer_sig_params = inspect.signature(Trainer.__init__).parameters
if "tokenizer" in trainer_sig_params:
    trainer_kwargs["tokenizer"] = tokenizer
elif "processing_class" in trainer_sig_params:
    trainer_kwargs["processing_class"] = tokenizer

trainer = Trainer(**trainer_kwargs)

print("Trainer initialized.")
print("Using argument name:", "evaluation_strategy" if "evaluation_strategy" in training_sig_params else "eval_strategy")
if "tokenizer" in trainer_sig_params:
    print("Trainer text processor argument: tokenizer")
elif "processing_class" in trainer_sig_params:
    print("Trainer text processor argument: processing_class")
else:
    print("Trainer text processor argument: none")

trainer.train()

val_metrics = trainer.evaluate(eval_dataset=val_dataset)
test_metrics = trainer.evaluate(eval_dataset=test_dataset)

print("Validation metrics:")
for k, v in val_metrics.items():
    print(f"  {k}: {v}")

print("\nTest metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v}")

ARTIFACT_DIR = "biomedbert_finetuned_mesh_multilabel"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

trainer.save_model(ARTIFACT_DIR)
tokenizer.save_pretrained(ARTIFACT_DIR)
training_args_path = os.path.join(ARTIFACT_DIR, "training_args.json")
if hasattr(training_args, "to_json_file"):
    training_args.to_json_file(training_args_path)
else:
    args_dict = (
        training_args.to_sanitized_dict()
        if hasattr(training_args, "to_sanitized_dict")
        else training_args.to_dict()
    )
    with open(training_args_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, ensure_ascii=False, indent=2)

with open(os.path.join(ARTIFACT_DIR, "multi_label_binarizer.pkl"), "wb") as f:
    pickle.dump(mlb, f)

with open(os.path.join(ARTIFACT_DIR, "mesh_classes.txt"), "w", encoding="utf-8") as f:
    for label in mlb.classes_:
        f.write(f"{label}\n")

with open(os.path.join(ARTIFACT_DIR, "threshold.txt"), "w", encoding="utf-8") as f:
    f.write(str(THRESHOLD))

print("saved artifacts to:", ARTIFACT_DIR)

sample_text = test_texts[0]

inputs = tokenizer(
    sample_text,
    truncation=True,
    padding=True,
    max_length=MAX_LEN,
    return_tensors="pt",
).to(trainer.model.device)

with torch.no_grad():
    logits = trainer.model(**inputs).logits[0].cpu()

probs = torch.sigmoid(logits).numpy()
pred_multi_hot = (probs >= THRESHOLD).astype(int)
pred_terms = mlb.inverse_transform(pred_multi_hot.reshape(1, -1))[0]

print("Abstract snippet:")
print(sample_text[:800] + "...")
print("\nPredicted term count:", len(pred_terms))
print("Predicted MeSH terms (first 30):", pred_terms[:30])

k = 20
top_idx = np.argsort(probs)[-k:][::-1]
print("\nTop probabilities:")
for i in top_idx:
    print(f"{mlb.classes_[i]}: {probs[i]:.4f}")