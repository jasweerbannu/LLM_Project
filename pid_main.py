# Step 1: Install Required Libraries
# Ensure necessary libraries are installed before running the script
# !pip install transformers datasets torch

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load and Prepare the Dataset
# Load cleaned dataset
file_path = 'Cleaned_PromptInjectionDataset.xlsx'
try:
    df_cleaned = pd.read_excel(file_path)  # Check if the file is accessible
    print(f"Dataset loaded successfully with {df_cleaned.shape[0]} rows.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Debug: Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df_cleaned.head())

# Split dataset into train and test
try:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_cleaned['text'], df_cleaned['label'], test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(train_texts)}, Test set size: {len(test_texts)}")
except KeyError as e:
    print(f"Error splitting dataset: Missing column {e}")
    raise

# Convert to Hugging Face Dataset format
try:
    train_data = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
    test_data = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
    print("Data successfully converted to Hugging Face Dataset format.")
except Exception as e:
    print(f"Error converting to Hugging Face Dataset format: {e}")
    raise

# Step 3: Tokenize the Data
# Load tokenizer
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

try:
    train_data = train_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)
    print("Tokenization completed.")
except Exception as e:
    print(f"Error during tokenization: {e}")
    raise

# Remove text column
try:
    train_data = train_data.remove_columns(["text"])
    test_data = test_data.remove_columns(["text"])
    print("Text column removed from datasets.")
except KeyError as e:
    print(f"Error removing text column: {e}")
    raise

# Set format for PyTorch
try:
    train_data = train_data.with_format("torch")
    test_data = test_data.with_format("torch")
    print("Datasets converted to PyTorch format.")
except Exception as e:
    print(f"Error setting dataset format: {e}")
    raise

# Step 4: Load Pretrained Model
try:
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    print("Pretrained model loaded successfully.")
except Exception as e:
    print(f"Error loading pretrained model: {e}")
    raise

# Step 5: Define Training Arguments
try:
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
    )
    print("Training arguments defined successfully.")
except Exception as e:
    print(f"Error defining training arguments: {e}")
    raise

# Step 6: Train the Model
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
    )
    print("Training initialized. Starting training...")
    trainer.train()
    print("Training completed successfully.")
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Step 7: Evaluate the Model
try:
    # Get predictions
    predictions = trainer.predict(test_data)
    preds = predictions.predictions.argmax(-1)  # Convert logits to predicted labels
    true_labels = test_data["label"]  # Actual labels

    # Calculate metrics
    print("Generating evaluation metrics...")
    print(f"Accuracy: {accuracy_score(true_labels, preds):.4f}")
    print("Classification Report:")
    print(classification_report(true_labels, preds, target_names=["Safe", "Unsafe"]))

except Exception as e:
    print(f"Error during evaluation metrics computation: {e}")
    raise

# Step 8: Save the Model
try:
    model.save_pretrained("./distilbert-prompt-injection")
    tokenizer.save_pretrained("./distilbert-prompt-injection")
    print("Model and tokenizer saved to './distilbert-prompt-injection'")
except Exception as e:
    print(f"Error saving model and tokenizer: {e}")
    raise
