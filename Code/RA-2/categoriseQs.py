import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define categories and their associated keywords/phrases
categories = {
    "Modal Verbs": ["can", "could", "would", "should", "may", "might", "will", "shall", "must"],
    "Interrogatives": ["who", "what", "when", "where", "why", "how", "which"],
    "Do/Does/Did Questions": ["do", "does", "did"],
    "Doubt/Clarification": [
        "is it possible", "do you mean", "are you saying", 
        "does this imply", "could you clarify", "what if", 
        "does that mean", "is there a reason"
    ],
    "Contextual Questions": [
        "is there", "will this", "does this", "should we", 
        "could this", "is this", "can this"
    ],
    "Exploration/Inquiry": ["explain", "define", "elaborate", "clarify", "give details about", "go into depth about"],
    "Comparison Questions": ["how does this compare", "is this better", "which one is better", "what's the difference"],
    "Hypothetical Scenarios": ["what if", "imagine if", "suppose", "assuming that", "let's say"],
    "Problem-Solving Questions": ["how can we", "what's the solution", "how do we fix", "how do we solve"],
    "Confirmation Questions": ["is it true", "is this correct", "am I right", "does this match"],
    "Reasoning/Justification": ["why is", "what's the reason", "how come", "why does this"],
    "Permission/Request": ["can I", "could you", "may I", "is it okay if"],
    "Instruction/Procedure": ["how do I", "what's the process", "what steps", "how to"]
}


# Detect if a comment is a question
def is_question(comment):
    # Basic heuristic: Check for a question mark first
    if "?" in comment:
        # If a question mark is present, check for keywords
        tokens = word_tokenize(comment.lower())
        for category, keywords in categories.items():
            if any(keyword in tokens for keyword in keywords):
                return True
    # If no question mark, assume it's not a question
    return False

# Categorize a question based on the categories
def categorize_question(comment):
    tokens = word_tokenize(comment.lower())
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in tokens:
                return category, keyword
    return "Uncategorized", None

# Process a comment and determine if it's a question or an answer
def process_comment(comment, q_count):
    if is_question(comment):
        category, keyword = categorize_question(comment)
        if category != "Uncategorized":
            q_count[category][keyword] += 1
        return {"type": "Q", "category": category, "body": comment}
    else:
        return {"type": "Answer", "body": comment}

# Process a JSON file and categorize comments
def process_json(file_path):
    # Load the JSON data
    with open(file_path, 'r') as file:
        data = json.load(file)

    categorized_data = {}
    q_count = defaultdict(lambda: defaultdict(int))

    # Process each repository
    for repo_url, issues in data.items():
        categorized_data[repo_url] = []

        # Process each issue
        for issue in issues:
            issue_url = issue["issue_url"]
            issue_body = issue["issue_body"]
            comments = issue["comments"]

            # Process issue comments
            processed_comments = []
            for comment in comments:
                processed_comment = process_comment(comment["body"], q_count)
                processed_comment["comment_url"] = comment["comment_url"]
                processed_comment["issue_url"] = issue_url  # Link to the issue
                processed_comment["user"] = comment["user"]
                processed_comment["created_at"] = comment["created_at"]
                processed_comment["reactions"] = comment.get("reactions", [])
                processed_comments.append(processed_comment)

            # Add processed comments under the issue
            categorized_data[repo_url].append({
                "issue_url": issue_url,
                "issue_body": issue_body,
                "comments": processed_comments
            })

    return categorized_data, q_count

# Save categorized data to a new JSON file
def save_categorized_data(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2)
    print(f"Categorized data saved to {output_file}")

# Display Q counts
def display_q_counts(q_count):
    print("\nQuestion Counts by Category and Sub-Category:")
    for category, keywords in q_count.items():
        print(f"\n{category}:")
        for keyword, count in keywords.items():
            print(f"  {keyword} = {count}")

# Main function
def main():
    input_file = "issuesWithComments.json"
    output_file = "categorizedCommentsV3.json"

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    try:
        categorized_data, q_count = process_json(input_file)
        save_categorized_data(categorized_data, output_file)
        display_q_counts(q_count)
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    main()
