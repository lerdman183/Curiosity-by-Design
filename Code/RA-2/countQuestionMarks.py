import json

def count_question_marks_in_json(file_path):
    """
    Counts the number of '?' in the 'issue_body' and in the 'body' field of 'comments' 
    in a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: A dictionary with the counts of '?' in 'issue_body' and 'comments' fields.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        # print(list(data.items())[:1])

        issue_body_question_marks = 0
        comments_question_marks = 0

        for repo, issues in data.items():
            for issue in issues:
                # Count '?' in issue_body
                issue_body = str(issue.get("issue_body", ""))
                issue_body_question_marks += issue_body.count("?")

                # Count '?' in comments' body fields
                comments = issue.get("comments", [])
                for comment in comments:
                    comment_body = str(comment.get("body", ""))
                    comments_question_marks += comment_body.count("?")

        return {
            "issue_body_question_marks": issue_body_question_marks,
            "comments_question_marks": comments_question_marks,
            "total_question_marks": issue_body_question_marks + comments_question_marks,
        }

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from {file_path}")
        return None

# Main execution
if __name__ == "__main__":
    file_path = "/Users/harshdarji/Documents/RA/RA-2/issuesWithComments.json"
    result = count_question_marks_in_json(file_path)
    
    if result:
        print("\nQuestion Mark Counts:")
        print(f"Issue Body: {result['issue_body_question_marks']}")
        print(f"Comments: {result['comments_question_marks']}")
        print(f"Total: {result['total_question_marks']}")
