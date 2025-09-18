# train_finbert.py
# This script trains a FinBERT model for financial sentiment analysis

print("Starting FinBERT training script...")

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os

def compute_metrics(p):
    """Computes and returns evaluation metrics."""
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='weighted', zero_division=0)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    """Main function to load data, fine-tune the FinBERT model, and save it."""
    # --- 1. Define Paths for Colab's Temporary Session Storage ---
    # This path points to the file you just uploaded.
    data_path = "/content/emotion_dataset.csv"
    # The final model will be saved here, in the temporary storage.
    output_dir = "/content/finbert_emotion_model"
    
    # --- 2. Load and Prepare the Dataset ---
    print(f"Loading dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("ERROR: 'emotion_dataset.csv' not found in the Colab session storage.")
        print("Please make sure you have uploaded the file using the folder icon on the left.")
        return

    print("Dataset loaded successfully. Preprocessing data...")
    df['text'] = df['Clean_Text'].fillna(df['Text'])
    df.dropna(subset=['text', 'Emotion'], inplace=True)
    
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['Emotion'])
    num_labels = len(label_encoder.classes_)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    
    print(f"Found {num_labels} unique emotions.")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # --- 3. Load Tokenizer and Model ---
    print("Loading FinBERT tokenizer and model...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True # ESSENTIAL for transfer learning
    )
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    # --- 4. Fine-Tune the Model ---
    print("Starting model fine-tuning...")
    print(f"Model will be saved to: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
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

    # --- 5. Save the Final Model ---
    print(f"Training complete. Saving final model to '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    main()