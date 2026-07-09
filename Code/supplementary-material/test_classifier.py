import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Specify your checkpoint directory
checkpoint_dir = "/content/checkpoint-1872"

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
model.eval()

# 3. (Optional) Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. Define an inference function with timing
def classify_requests(request_texts, batch_size=16, max_length=256):
    """
    request_texts: List[str] of user requests
    Returns: Tuple of (results, per_query_times, avg_time)
        - results: List of (pred_label_index, pred_probabilities) tuples
        - per_query_times: List of time taken per query in seconds
        - avg_time: Average time per query in seconds
    """
    all_logits = []
    per_query_times = []

    # Process in batches
    for i in range(0, len(request_texts), batch_size):
        batch = request_texts[i : i + batch_size]
        enc = tokenizer(batch,
                        truncation=True,
                        padding=True,
                        max_length=max_length,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits  # shape (batch_size, num_labels)
        end_time = time.time()

        # Calculate time for the batch and per query
        batch_time = end_time - start_time
        batch_size_actual = len(batch)  # Handle last batch with fewer items
        per_query_time = batch_time / batch_size_actual

        # Append per-query times for each query in the batch
        per_query_times.extend([per_query_time] * batch_size_actual)
        all_logits.append(logits.cpu())

    # Concatenate logits
    all_logits = torch.cat(all_logits, dim=0)  # (N, num_labels)

    # Convert to probabilities and predictions
    probs = torch.softmax(all_logits, dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    results = list(zip(preds, probs))

    # Calculate average time
    avg_time = sum(per_query_times) / len(per_query_times) if per_query_times else 0

    return results, per_query_times, avg_time

# 5. Example usage
if __name__ == "__main__":
    test_prompts = [
        """
        "function fetchUserData(userId) {
          // TODO: fetch user data from API
        return userData;
        }
        let data = fetchUserData(1);"
        """,
        """
        "<html>
    <head>
        <title>My Page</title>
    </head>
    <body>
        <h1>Welcome to my page
        <p>This is a paragraph."
        """,
        """
        I need assistance with debugging.
        """,
        """
        I'm confused about data structures.
        """,
        """
        Can someone explain how to optimize my code?
        """,
        """
        "import requests

    # TODO: Ensure headers are set for API request
    response = requests.get('https://api.example.com/data')
    print(response)"
        """,
        """
        function sendRequest(url) {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url);
      // TODO: Add onload handler
    }
        """,
        """
        function fetchData(url) {
      // TODO: Implement fetch logic
      return;
    }

    fetchData('http://example.com')
        """,
        """
        def connect_to_db():
      connection = None
      # TODO: Establish database connection
      return connection
        """,
        """
        "class Animal:
      def __init__(self, name):
          self.name = name
      # TODO: Create a method to make sound"
        """
    ]

    # Run classification with timing
    results, per_query_times, avg_time = classify_requests(test_prompts)

    # Print results
    print(f"Number of queries: {len(results)}")
    print(f"Average time per query: {avg_time:.6f} seconds")
    print("\nDetailed results:")
    for i, (text, (pred_idx, prob_dist), query_time) in enumerate(zip(test_prompts, results, per_query_times)):
        print(f"\nQuery {i+1}:")
        print(f"Input: {text.strip()}")
        print(f"Predicted class index: {pred_idx} (original label = {pred_idx+1})")
        print(f"Probabilities: {prob_dist.round(3)}")
        print(f"Time taken: {query_time:.6f} seconds")