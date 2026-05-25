import json
import random

def sample_issues_with_questions(json_file_path, sample_size=50):
    try:
        # Load JSON data from the file
        with open(json_file_path, "r") as file:
            data = json.load(file)

        # Collect all issues containing at least one comment with "type": "Q"
        issues_with_questions = []
        for repo, issues in data.items():
            for issue in issues:
                # Check if any comment in the issue has "type": "Q"
                if any(comment.get("type") == "Q" for comment in issue["comments"]):
                    issues_with_questions.append(issue)

        # Randomly sample the required number of issues
        if len(issues_with_questions) < sample_size:
            print(f"Only {len(issues_with_questions)} issues with 'type: Q' found. Sampling all available.")
            sampled_issues = issues_with_questions
        else:
            sampled_issues = random.sample(issues_with_questions, sample_size)

        # Save the sampled data to a new JSON file
        output_file_path = "sampled_issues.json"
        with open(output_file_path, "w") as output_file:
            json.dump(sampled_issues, output_file, indent=2)

        print(f"Successfully sampled {len(sampled_issues)} issues with questions.")
        print(f"Sampled data saved to {output_file_path}")

    except Exception as e:
        print("An error occurred:", str(e))


# Specify the path to your JSON file
json_file_path = "categorizedCommentsV3.json"  # Update with the actual path to your JSON file
sample_issues_with_questions(json_file_path)
