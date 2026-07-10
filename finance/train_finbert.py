# train_finbert.py
# Fine-tunes a FinBERT model for emotion/financial sentiment classification.
#
# Run from the project root, e.g.:
#     python -m finance.train_finbert --data Data/emotion_dataset.csv --output finbert_emotion_model

import argparse

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def compute_metrics(p):
    """Compute accuracy and weighted F1."""
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average="weighted", zero_division=0)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT for sentiment.")
    parser.add_argument(
        "--data",
        default="Data/emotion_dataset.csv",
        help="Path to the training CSV (needs Text/Clean_Text and Emotion columns).",
    )
    parser.add_argument(
        "--output",
        default="finbert_emotion_model",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--model-name",
        default="yiyanghkust/finbert-pretrain",
        help="Base model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from: {args.data}")
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"ERROR: '{args.data}' not found.")
        return

    print("Dataset loaded successfully. Preprocessing data...")
    df["text"] = df["Clean_Text"].fillna(df["Text"]) if "Clean_Text" in df else df["Text"]
    df.dropna(subset=["text", "Emotion"], inplace=True)

    label_encoder = LabelEncoder()
    df["labels"] = label_encoder.fit_transform(df["Emotion"])
    num_labels = len(label_encoder.classes_)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

    print(f"Found {num_labels} unique emotions.")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["labels"]
    )
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print("Loading FinBERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # ESSENTIAL for transfer learning
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    print(f"Starting model fine-tuning... (output: {args.output})")
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(f"Training complete. Saving final model to '{args.output}'...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Model and tokenizer saved successfully.")


if __name__ == "__main__":
    main()
