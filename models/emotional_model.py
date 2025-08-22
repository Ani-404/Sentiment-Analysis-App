# train_bert.py

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

def compute_metrics(p):
    """
    Computes and returns evaluation metrics.
    
    Args:
        p (EvalPrediction): A named tuple with predictions and label_ids.
        
    Returns:
        dict: A dictionary containing the accuracy and F1 score.
    """
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1}

def main():
    """
    Main function to load data, fine-tune the BERT model, and save it.
    """
    # --- 1. Load and Prepare the Dataset ---
    print("Loading and preparing the dataset...")
    
    # Load the dataset from the CSV file
    df = pd.read_csv('emotion_dataset.csv')
    
    # Use 'Clean_Text' if available, otherwise fall back to 'Text'
    df['text'] = df['Clean_Text'].fillna(df['Text'])
    
    # Drop rows where text is still missing (if any)
    df.dropna(subset=['text', 'Emotion'], inplace=True)
    
    # Encode the string labels into integers
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['Emotion'])
    
    # Get the number of unique labels
    num_labels = len(label_encoder.classes_)
    
    # Create mappings from ID to label and vice-versa
    # This is crucial for interpreting the model's output later
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
    
    print(f"Found {num_labels} unique emotions.")

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
    
    # Convert pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # --- 2. Load Tokenizer and Preprocess Data ---
    print("Loading tokenizer and preprocessing data...")
    
    # Define the model checkpoint to use
    model_name = "distilbert-base-uncased"
    
    # Load the tokenizer associated with the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        """Tokenizes the text data."""
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    # Apply the tokenization to the datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    # --- 3. Load and Configure the Model ---
    print("Loading pre-trained model...")
    
    # Load the pre-trained model for sequence classification
    # Configure it with the correct number of labels and the label mappings
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # --- 4. Fine-Tune the Model ---
    print("Starting model fine-tuning...")
    
    # Define the directory where the model will be saved
    output_dir = "./sentiment_model_bert"

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,              # Total number of training epochs
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=16,   # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=100,               # Log every X updates steps
        evaluation_strategy="epoch",     # Evaluate at the end of each epoch
        save_strategy="epoch",           # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True,     # Load the best model found during training
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )

    # Start the training process
    trainer.train()

    # --- 5. Save the Fine-Tuned Model and Tokenizer ---
    print(f"Training complete. Saving model and tokenizer to '{output_dir}'...")
    
    # Save the final model and tokenizer to the specified directory
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    main()
