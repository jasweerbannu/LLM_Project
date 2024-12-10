
# Machine Learning Prompt Injection Project

This repository contains a machine learning project designed to classify queries as safe or unsafe and provide meaningful educational responses using OpenAI's ChatGPT API. It leverages Hugging Face Transformers, PyTorch, and Flask for model training, evaluation, and API deployment.

## Features
- **Classification Models**: Uses DistilBERT and a fine-tuned custom model for sequence classification.
- **Semantic Similarity**: Implements Sentence Transformers for context-aware query evaluation.
- **ChatGPT Integration**: Employs OpenAI's ChatGPT API for generating tailored responses.
- **Flask API**: Provides a REST API for evaluating and responding to queries.
- **Data Preparation**: Includes scripts for tokenizing, splitting, and augmenting datasets.

## Requirements
Ensure the following libraries are installed before running the project:

flask
torch
transformers
openai
sentence-transformers
pandas
scikit-learn
datasets
spacy
scipy
xlrd


Install dependencies using:

pip install -r requirements.txt


## Setup and Usage

### 1. Clone the Repository

git clone https://github.com/jasweerbannu/LLM_Project.git
cd LLM_Project


### 2. Prepare the Dataset
- Place the cleaned dataset in the project directory as `Cleaned_dataset.xslx` & `Cleaned_PromptInjectionDataset.xlsx`.

### 3. Train the Models
Run the training scripts to fine-tune DistilBERT and the custom classification model:

python main.py
python pid_main.py

### 4. Start the Flask API
Run the Flask application to deploy the API:

python app.py


The API will be accessible at `http://localhost:5000`.

### 5. Query the API
Send a POST request to `/evaluate` with a JSON body:
```json
{
    "query": "How do I calculate the area of a triangle?"
}
```

The API will return a safety evaluation and a detailed response.

## Project Structure
- `app.py`: Flask application for the REST API.
- `main.py`: Script for model training and evaluation of safe or unsafe query detection model.
- `requirements.txt`: List of required libraries.
- `Cleaned_PromptInjectionDataset.xlsx`: Preprocessed dataset for prompt injection or not querires for training.
- `Cleaned_dataset.xslx` : Preprocessed dataset for safe and unsafe queries for training.
- `trained_model`: Directory for saving trained safe or unsafe model.
- `distilbert-prompt-injection`: Directory for saving fine-tuned DistilBERT model.

## Model Outputs
- Safety Evaluation Classifies the query as safe or unsafe.
- Educational Response Generates a thoughtful response to safe queries or guidance for unsafe queries.

## Contributions
Feel free to fork this repository and submit pull requests with enhancements or fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
