import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
BASE_MODEL = "google/gemma-3-1b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Function to load model and tokenizer ---
def load_model_and_tokenizer():
    print("Loading tokenizer and model...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token  # Important for padding

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32,
        device_map="auto" if DEVICE.startswith("cuda") else None,
    )
    model.to(DEVICE)
    model.eval()

    end_time = time.time()
    print(f"Model and tokenizer loaded in {end_time - start_time:.2f} seconds")
    return tokenizer, model

# --- Test data ---
test_data = [
    {
        "prompt": """
        “Extra context:
            User question: function fetchUserData(userId) { // TODO: fetch user data from API return userData; } let data = fetchUserData(1);
            AI clarification question: Thanks for sharing the code! It looks like you're trying to fetch user data from an API. Could you clarify:
            What data should the function return? (e.g., specific fields like name, email)
            What should happen if the API call fails? (e.g., return an error message, throw an exception)
            User follow-up interaction: I want the function to return a JSON object with name, email, and id fields.”
        """,
        "label": 2
    },
    {
        "prompt": """
        Extra context:
User Question:
xml
<html>
<head>
    <title>My Page</title>
</head>
<body>
    <h1>Welcome to my page
    <p>This is a paragraph.

AI clarification question: What is the primary purpose or goal of this page?
User follow up response: I'm creating a blog page to display my posts.
        """,
        "label": 2
    },
    {
        "prompt": """
        Extra context:
User Question: I need assistance with debugging.
AI clarification question: What programming language are you using? (e.g., Python, JavaScript, C++, etc.)
User follow up response: I'm debugging a Python script, and I'm getting an 'IndexError: list index out of range'.
        """,
        "label": 2
    },
    {
        "prompt": """
        Extra context:
        User Question: I'm confused about data structures.
AI clarification question: Could you tell me what area of data structures you're finding confusing? For example, are you unsure about what data structure to use, how to implement one, or something else?
User follow up response: I'm confused about how to choose between a hash table and an array for storing key-value pairs.
        """,
        "label": 1
    },
    {
        "prompt": """
        “Extra context:
User Question: Can someone explain how to optimize my code?
AI clarification question: What language are you using for your code, and what specific part of it are you finding slow or inefficient? (e.g., loops, data processing, etc.)
User follow up response: I'm using C++, and my loop for processing a large dataset is too slow.
""",
        "label": 1
    },
    {
        "prompt": """
        Extra context:
User Question:
python
import requests

# TODO: Ensure headers are set for API request
response = requests.get('https://api.example.com/data')
print(response)

AI clarification question: Please clarify what you want to achieve with this code snippet. What API endpoint are you trying to access and what outcome are you hoping to get from the API response?
User follow up response: I'm trying to fetch user data from a custom API, expecting a JSON response with user details.
""",
        "label": 1
    },
    {
        "prompt": """
        Extra context:
User Question:
function sendRequest(url) {...}

AI clarification question: What is the intended purpose of the function sendRequest within the context of web development? Specifically, what action or data are you hoping to send to the URL specified in the function?
User follow up response: I want to fetch JSON data from the URL and log it to the console.
""",
        "label": 1
    },
    {
        "prompt": """
        Extra context:
User Question:

function fetchData(url) {
// TODO: Implement fetch logic
 	return;
}
fetchData('http://example.com')

AI clarification question: What is the intended purpose of the fetchData function, and what data or action should the function perform after receiving a URL? For example, do you want to fetch data from a specific API, or something else entirely?
User follow up response: I want to fetch JSON data from a REST API and return it.

        """,
        "label": 1
    },
    {
        "prompt": """
        Extra context:

User Question:

def connect_to_db():
	connection = None
# TODO: Establish database connection
return connection

AI clarification question: What type of database are you trying to connect to, and what specific error are you expecting to encounter when you try to connect? For example, are you getting a connection error, or is it just a general feeling that the connection isn't working?
User follow up interaction: I'm trying to connect to a PostgreSQL database and getting a 'DatabaseError: connection timeout' error.
        """,
        "label": 1
    },
    {
        "prompt": """
        “Extra context:

User question:



class Animal:
def init(self, name):
self.name = name
# TODO: Create a method to make sound

AI clarification question:
Thanks for sharing the code! It looks like you want to add a method to make animal sounds. Could you clarify:
What specific sounds should the method produce? (e.g., different sounds for different animals)
How should the method behave? (e.g., print the sound, return it as a string)
User follow up interaction: I want the method to return "Woof" for a dog and "Meow" for a cat, based on the name.”

        """,
        "label": 2
    }
]

# --- Reasoning generation function with timing ---
def generate_reasoning(prompt: str, label: int, tokenizer, model, max_new_tokens: int = 256):
    """
    Generate reasoning for a prompt and measure response time.
    Returns: Tuple of (reasoning, response_time)
    """
    instr = (
        f"<start_of_turn>user\n"
        f"{prompt}\n"
        "Keep your response to the point and concise."
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    tokens = tokenizer(instr, return_tensors="pt", padding=True)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    # Measure generation time
    start_time = time.time()
    with torch.inference_mode():
        outputs = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        logits = outputs.logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError("Model produced NaN or Inf in logits before generation.")

        out = model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=100,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    end_time = time.time()
    response_time = end_time - start_time

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    reasoning = full_text.split("<start_of_turn>model\n")[-1].strip()
    return reasoning, response_time

# --- Main execution ---
if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Process test data and collect results with timing
    results = []
    response_times = []
    print("\nProcessing test queries...")
    for i, ex in enumerate(test_data):
        try:
            reasoning, response_time = generate_reasoning(
                ex["prompt"], ex["label"], tokenizer, model
            )
            response_times.append(response_time)
        except RuntimeError as e:
            reasoning = f"Error during generation: {str(e)}"
            response_time = 0.0  # Mark as 0 for errors
            response_times.append(response_time)

        print(f"─{'─' * 39}")
        print(f"Query {i+1}:")
        print(f"Prompt:  {ex['prompt'][:100]}...")  # Truncate for readability
        print(f"Label:   {ex['label']}")
        print(f"Reason:  {reasoning[:100]}...")  # Truncate for readability
        print(f"Time:    {response_time:.4f} seconds")

        results.append({
            "prompt": ex["prompt"],
            "response": reasoning,
            "response_time": response_time,
            "label": ex["label"]
        })

    # Calculate and display average time
    avg_time = sum(response_times) / len(response_times) if response_times else 0
    print(f"\nSummary:")
    print(f"Number of queries: {len(results)}")
    print(f"Average response time: {avg_time:.4f} seconds")
    print(f"Total processing time: {sum(response_times):.4f} seconds")

    # Save results to JSON
    output_file = "results_gemma3-1b-it-raw-11_with_timing.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} entries to {output_file}")