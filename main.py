import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import spacy

# Step 1: Load Cleaned Dataset
output_path = "cleaned_dataset.xlsx"

print("Loading the cleaned dataset...")
df = pd.read_excel(output_path)
print(f"Loaded dataset shape: {df.shape}")

# Rename columns for compatibility with Hugging Face format
df = df.rename(columns={"title": "text", "label": "labels"})

# Ensure the labels are integers
df['labels'] = df['labels'].astype(int)

# Step 2: Split Dataset into Training and Testing Sets
print("Splitting the dataset into training and testing sets...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['labels'], test_size=0.2, random_state=42
)
print(f"Training set size: {len(train_texts)}, Testing set size: {len(test_texts)}")

# Step 3: Initialize Tokenizer
print("Initializing the tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Tokenizes the input text for the classification model
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Convert training and testing data into Hugging Face dataset format
print("Tokenizing training and testing data...")
train_data = Dataset.from_dict({"text": train_texts, "labels": train_labels})
test_data = Dataset.from_dict({"text": test_texts, "labels": test_labels})

# Apply tokenization to both datasets
train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)
print(f"Tokenization completed. Example tokenized input: {train_data[0]}")

# Step 4: Add Dependency Features via spaCy
print("Enhancing with linguistic features (including dependencies)...")
nlp = spacy.load("en_core_web_sm")

def extract_dependency_features(doc):
    """
    Extract features based on dependency parsing.
    Returns counts of key dependency types and other relevant structures.
    """
    dep_counts = {
        'ROOT': 0,
        'nsubj': 0,
        'aux': 0,
        'advmod': 0,
        'other': 0  # Catch-all for non-specific dependencies
    }
    for token in doc:
        if token.dep_ in dep_counts:
            dep_counts[token.dep_] += 1
        else:
            dep_counts['other'] += 1
    return dep_counts

def preprocess_with_features(dataset):
    def add_features(example):
        doc = nlp(example['text'])
        
        # Extract entity count and query length
        example['entity_count'] = len(doc.ents)
        example['query_length'] = len(doc)
        
        # Extract dependency structure counts
        dep_counts = extract_dependency_features(doc)
        example.update(dep_counts)  # Add dependency counts as features
        
        return example

    # Apply feature extraction to the dataset
    return dataset.map(add_features)

# Preprocess datasets with features
train_data = preprocess_with_features(train_data)
test_data = preprocess_with_features(test_data)
print("Feature extraction completed. Example enriched input:", train_data[0])

# Step 5: Load the Pre-trained Model
print("Loading the DistilBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print("Model loaded successfully.")

# Step 6: Configure Training Arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints
    eval_strategy="epoch",  # Evaluate the model at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Adjust batch size per device
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate a larger batch size
    num_train_epochs=3,  # Number of training epochs
    weight_decay=0.01,  # Regularization to avoid overfitting
    logging_dir="./logs",  # Directory for logging training metrics
    save_strategy="epoch",  # Save model at the end of each epoch
    save_total_limit=1,  # Keep only the best checkpoint
    load_best_model_at_end=True,  # Automatically load the best model
    metric_for_best_model="eval_loss",  # Optimize for evaluation loss
    greater_is_better=False  # Lower evaluation loss is better
)
print("Training arguments configured.")

# Step 7: Define Evaluation Metrics
def compute_metrics(eval_pred):
    # Compute accuracy, precision, recall, and F1-score
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Step 8: Initialize the Trainer
print("Setting up the Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("Trainer initialized.")

# Step 9: Train the Model
print("Starting model training...")
trainer.train()
print("Model training completed.")

# Step 10: Evaluate the Model
print("Evaluating the model on the testing set...")
results = trainer.evaluate()
print("Evaluation Results:", results)

# Step 11: Save the Model and Tokenizer
print("Saving the trained model and tokenizer...")
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("Model and tokenizer saved to './trained_model'.")
