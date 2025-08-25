# train_bert_emotion_model

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
import os # Import the 'os' module to handle file paths

def compute_metrics(p):
    """
    Computes and returns evaluation metrics.
    """
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='weighted', zero_division=0)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    """
    Main function to load data, fine-tune the BERT model, and save it.
    """
    # --- 1. Load and Prepare the Dataset ---
    print("Loading and preparing the dataset...")
    
    try:
        # Assuming the script is in 'models' folder, look one level up for the data
        df = pd.read_csv(r"C:\Users\anime\OneDrive\Desktop\Sentiment-Analysis-App\Data\emotion_dataset.csv")
    except FileNotFoundError:
        print("Error: 'emotion_dataset.csv' not found. Make sure it is in the main project directory.")
        return

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

    # --- 2. Load Tokenizer and Preprocess Data ---
    print("Loading tokenizer and preprocessing data...")
    
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    # --- 3. Load and Configure the Model ---
    print("Loading pre-trained model...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # --- 4. Fine-Tune the Model ---
    print("Starting model fine-tuning...")
    
    # --- KEY CHANGE: Save the model in the SAME directory as this script ---
    # This gets the absolute path to the folder containing this script.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "sentiment_model_tiny")
    print(f"Model will be saved to: {output_dir}")
    # --- End of Change ---

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # --- 5. Save and Verify the Model ---
    print(f"Training complete. Saving model and tokenizer to '{output_dir}'...")
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Model and tokenizer saved successfully!")
    
    # --- NEW: Verify that the files were created ---
    try:
        saved_files = os.listdir(output_dir)
        print("\n--- VERIFICATION ---")
        print(f"Successfully found the following files in '{output_dir}':")
        for file_name in saved_files:
            print(f"- {file_name}")
        print("--------------------")
    except Exception as e:
        print(f"Could not verify saved files. Error: {e}")
    # --- End of New Section ---

if __name__ == "__main__":
    main()

