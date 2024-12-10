from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openai
from sentence_transformers import SentenceTransformer, util  # For semantic similarity

# OpenAI API Key Configuration
openai.api_key = "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"  # Replace with your API key

# Initialize Flask app
app = Flask(__name__)

# Load models and tokenizers
print("Loading models and tokenizers...")
trained_model_path = "./trained_model"
distilbert_model_path = "./distilbert-prompt-injection"

trained_tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
trained_model = AutoModelForSequenceClassification.from_pretrained(trained_model_path)

distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_path)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_path)

print("Models loaded successfully.")

# Load sentence embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and efficient

# Store conversation history
conversation_history = []

def get_prediction(query, model, tokenizer):
    """
    Pass the query through the given model and return True (safe) or False (unsafe).
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class == 0  # Assuming 0 represents 'safe'


def is_related_to_previous(query, history):
    """
    Check if the new query is related to the previous conversation using semantic similarity
    and clarification keywords.
    """
    if not history:
        return False  # No history means no related context

    # Combine history into a single context string
    previous_context = " ".join([item["query"] for item in history])

    # Compute semantic similarity
    embeddings = embedding_model.encode([previous_context, query], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    # Clarification keywords
    clarification_keywords = ["tell me more", "explain", "any other", "clarify", "what else", "expand"]

    # Check for explicit clarification phrases in the query
    if any(keyword in query.lower() for keyword in clarification_keywords):
        return True

    # Adjust threshold for similarity
    return similarity > 0.6  # Lower threshold for better contextual understanding


# API call to ChatGPT
def chatgpt_api_call(prompt):
    """
    Use the OpenAI ChatGPT API to generate a response for the given prompt.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Specify the ChatGPT model
            messages=[
                {"role": "system", "content": "You are an intelligent educational assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error in generating response: {str(e)}"

# Handle safe query
def handle_safe_query_with_chatgpt(query):
    """
    Generate a response for safe queries using the safe query prompt via ChatGPT API call.
    """
    prompt = f"""
    The student query requires a thoughtful and educational response. Tailor your reply based on the nature of the query:
    - Step-by-Step Guidance: Break the problem into manageable steps to help the student understand the process without providing the final answer.
    - In-Depth Explanation: Provide a comprehensive explanation of the topic, using examples or analogies to enhance clarity.
    - Clarification: Focus on resolving the specific doubt in the query with a precise and concise explanation.

    Query: {query}
    """
    detailed_response = chatgpt_api_call(prompt)
    return format_response_to_html(detailed_response)

# Handle unsafe query
def handle_unsafe_query_with_chatgpt(query):
    """
    Generate a response for unsafe queries using the unsafe query prompt via ChatGPT API call.
    """
    initial_message = (
        "<strong>It seems you're seeking a direct answer.</strong> To foster deeper understanding and promote learning, "
        "let me guide you step by step:"
    )
    prompt = f"""
    This query appears to seek a direct answer. To foster deeper understanding and promote learning:
    - Decompose the Problem: Break down the query into clear, logical steps that guide the student.
    - Engage with Explanations: Provide thoughtful hints or illustrative examples to encourage active engagement.
    - Emphasize Reasoning: Avoid stating the final solution but focus on the reasoning and principles to help the student independently arrive at the answer.

    Query: {query}
    """
    detailed_response = chatgpt_api_call(prompt)
    return initial_message + "<br><br>" + format_response_to_html(detailed_response)

# Format response dynamically into HTML
def format_response_to_html(response_text):
    html_response = ""
    lines = response_text.split("\n")  # Split the response into lines

    for line in lines:
        line = line.strip()
        if line.startswith("- "):  # Bullet points
            html_response += f"<li>{line[2:].strip()}</li>"
        elif line.startswith("1.") or line.startswith("2.") or line.startswith("3."):  # Numbered lists
            html_response += f"<li>{line}</li>"
        elif ":" in line:  # Treat as a subheading
            parts = line.split(":", 1)
            html_response += f"<strong>{parts[0]}:</strong> {parts[1].strip()}<br>"
        else:  # Plain text
            html_response += f"{line}<br>"

    if "<li>" in html_response:
        html_response = html_response.replace("<li>", "<ul><li>", 1).replace("</li>", "</li></ul>", 1)
    return html_response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    query = data.get('query', '').strip()

    # Check if the query is related to the ongoing conversation
    related = is_related_to_previous(query, conversation_history)

    if related:
        # Append to conversation context
        conversation_history.append({"query": query})
        full_prompt = "\n".join(
            [f"User: {item['query']}\nAssistant: {item.get('response', '')}" for item in conversation_history]
        ) + f"\nUser: {query}"  # Add the current query at the end
    else:
        # Reset conversation context
        conversation_history.clear()
        conversation_history.append({"query": query})
        full_prompt = f"User: {query}"  # Treat as a fresh query

    # Model evaluations
    trained_safe = get_prediction(query, trained_model, trained_tokenizer)
    distilbert_safe = get_prediction(query, distilbert_model, distilbert_tokenizer)
    is_safe = trained_safe and distilbert_safe

    # Generate response
    if is_safe:
        response = handle_safe_query_with_chatgpt(full_prompt)
    else:
        response = handle_unsafe_query_with_chatgpt(full_prompt)

    # Update conversation context with response
    conversation_history[-1]["response"] = response

    return jsonify({"query": query, "is_safe": is_safe, "response": response})

if __name__ == '__main__':
    app.run(debug=True)
