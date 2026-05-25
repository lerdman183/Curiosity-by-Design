import re
import os
import openai
import json
import time
import random
openai.api_key = "your api key"

# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.error.RateLimitError, openai.error.ServiceUnavailableError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

@retry_with_exponential_backoff
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def process_hunk_line(line):
    pattern = r'^@@.*?@@'  # Regular expression pattern to match the line starting with @@
    match = re.match(pattern, line)
    if match:
        line = line[match.end():]  # Keep the content after '@@'
        return line
    elif line.startswith('-'):
        return line[1:]  # Keep the content after '-'
    elif line.startswith('+'):
        return ""  # Remove the line with '+'
    return line  # Keep other lines as is

def process_hunk_key(file_path):
    #result = []
    #comments = []
    cnt = 0
    with open(file_path, 'r') as file, open('ans_code_refinement.txt', 'a') as fp:
        for line in file:
            cnt += 1
            data = json.loads(line)
            hunk_content = data['hunk']
            comment = data['comment']
            #comments.append(comment)
            processed_hunk = "\n".join(process_hunk_line(line) for line in hunk_content.split("\n"))
            new_prompt = '''Suppose you are a code reviewer, you will now refine the given code following the reviewed suggestion, for example:
Example: given the original code: 
def main(args):
         backward_time.append(t2 - t1)
         print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                    format(epoch, forward_time[-1], backward_time[-1]))
         cross_entropy(logits[val_idx], labels[val_idx])
         val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
         print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
               format(train_acc, loss.item(), val_acc, val_loss.item()))
Given the review suggestion:
 "`F.cross_entropy`? Also, isn't `tran_acc` required in L123?".
You should output the refined code following the suggestion: 
def main(args):
         backward_time.append(t2 - t1)
         print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
               format(epoch, forward_time[-1], backward_time[-1]))
         train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
         val_loss = F.cross_entropy(logits[val_idx], labels[val_idx]);
         val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
         print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
               format(train_acc, loss.item(), val_acc, val_loss.item()))
Now, give you a new original code:
            '''
            new_prompt += processed_hunk
            new_prompt += " and given the review suggestion: "
            new_prompt += comment
            new_prompt += " Please output the refined code following the suggestion, note that you should only output the code without other explanations."
            messages = [
                {"role": "system", "content": "Suppose you are a code reviewer, please revise the original code following the revised suggestion:"},
                {"role": "user", "content": new_prompt}
            ]
            response = completion_with_backoff(model="gpt-3.5-turbo", messages=messages, max_tokens=1000)
            # print(response["choices"][0]["message"]["content"])
            fp.write(str(cnt) + '\n')
            try:
                fp.write(response["choices"][0]["message"]["content"] + '\n')
            except:
                continue
            #result.append(processed_hunk)
    #return result, comments

file_path = 'ref-test.jsonl'
process_hunk_key(file_path)
