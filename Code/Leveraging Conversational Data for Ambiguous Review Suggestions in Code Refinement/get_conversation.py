

import requests
import json


# Your GitHub personal access token, replace with your own
headers = {
    'Authorization': 'your token',
}

# Function to get comments from a pull request
def get_comments(repo, pr_id):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_id}/comments"
    response = requests.get(url, headers=headers)
    conversation = []
    if response.status_code == 200:
        comments = response.json()
        for comment in comments:
            user = comment.get('user')
            if user and 'login' in user and 'body' in comment:
                conversation.append({
                    'user': user['login'],
                    'comment': comment['body']
                })
    else:
        print(f"Request failed with status code {response.status_code}")
    return conversation

# Function to get reviews from a pull request
def get_reviews(repo, pr_id):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_id}/reviews"
    response = requests.get(url, headers=headers)
    reviews = []
    if response.status_code == 200:
        reviews_data = response.json()
        for review in reviews_data:
            user = review.get('user')
            if user and 'login' in user and 'state' in review:
                reviews.append({
                    'user': user['login'],
                    'state': review['state']
                })
    else:
        print(f"Request failed with status code {response.status_code}")
    return reviews

import base64

def get_context_codes(repo, pr_id):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_id}/comments"
    response = requests.get(url, headers=headers)
    conversation = []
    if response.status_code == 200:
        comments = response.json()
        for comment in comments:
            user = comment.get('user')
            if user and 'login' in user and 'body' in comment:
                # Extract context information
                context_info = {
                    'path': comment.get('path'),
                    'position': comment.get('original_position'),
                    'commit_id': comment.get('original_commit_id')
                }
                # Get the file content
                file_url = f"https://api.github.com/repos/{repo}/contents/{context_info['path']}?ref={context_info['commit_id']}"
                file_response = requests.get(file_url, headers=headers)
                if file_response.status_code == 200:
                    file_content = file_response.json().get('content')
                    if file_content:
                        # The file content is base64 encoded, so we need to decode it
                        file_content = base64.b64decode(file_content).decode('utf-8')
                    else:
                        file_content = 'File content not found'
                else:
                    print(f"Request for file content failed with status code {file_response.status_code}")
                    file_content = 'Request for file content failed'
                context_info['file_content'] = file_content

                conversation.append({
                    'user': user['login'],
                    'comment': comment['body'],
                    'context': context_info
                })
    else:
        print(f"Request failed with status code {response.status_code}")
    return conversation

# Read JSONL file and process each line
with open('ref-test.jsonl', 'r') as f_in:
    lines = f_in.readlines()
    data_out = []
    for line in lines:  # process only the first 200 lines
        json_obj = json.loads(line)
        repo = json_obj.get('repo')
        pr_id = json_obj.get('ghid')
        if repo and pr_id:
            conversation = get_comments(repo, pr_id)
            data_out.append({
                'repo': repo,
                'pr_id': pr_id,
                'conversation': conversation
            })

# Write output to a new JSON file
with open('output_review_conversation13000.json', 'w') as f_out:
    json.dump(data_out, f_out)
