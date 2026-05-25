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
        errors: tuple = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError),
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

def load_data_from_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def build_conversation_dict(data_list):
    conversation_dict = {}
    for data in data_list:
        for item in data:
            key = (item['repo'], item['pr_id'])
            conversation_dict[key] = item['conversation']
    return conversation_dict
# Provide a list of file paths that you want to read
file_paths = ['output_review_conversation1000.json', 'output_review_conversation2000.json', 'output_review_conversation3000.json', 'output_review_conversation4000.json', 'output_review_conversation5000.json', 'output_review_conversation6000.json', 'output_review_conversation7000.json', 'output_review_conversation8000.json','output_review_conversation9000.json','output_review_conversation10000.json','output_review_conversation11000.json','output_review_conversation12000.json','output_review_conversation13000.json' ]

# Load data from each file and append it to the data_list
data_list = []
for file_path in file_paths:
    data = load_data_from_json_file(file_path)
    data_list.append(data)

# Build the conversation dictionary by merging all data from the data_list
conversation_dict = build_conversation_dict(data_list)

def process_hunk_key(file_path):
    #result = []
    #comments = []
    cnt = 0
    with open(file_path, 'r') as file, open('ans_code_refinement_ask_conversation.txt', 'a') as fp:
        for line in file:
            cnt += 1
            data = json.loads(line)
            hunk_content = data['hunk']
            comment = data['comment']
            #comments.append(comment)
            processed_hunk = "\n".join(process_hunk_line(line) for line in hunk_content.split("\n"))
            # give more examples that need the conversation information
            new_prompt = '''Suppose you are a code reviewer, you will now refine the given code following the reviewed suggestion. Note that you can exploit the external knowledge if you think the current review suggestion is not informative or confused. 
In concrete, you will first answer the question whether to exploit the external knowledge.
If Yes, you will acquire the conversation information. Otherwise, you can directly output the answer. For example:
given the original code: 
func (ai *AttendedInstaller) initializeUI() (err error) {
 	ai.keyboard, err = speakuputils.CreateVirtualKeyboard()
 	if err != nil {
 		logger.Log.Warnf("Failed to initialize virtual keyboard via uinput")
 	}
 	const osReleaseFile = "/etc/os-release"
 	ai.backdropStyle = tview.Theme{
Given the review suggestion:
Is this error fatal?
If you feel the current comment is not informative enough for revising the code, you can choose to exploit the conversation information, otherwise you can directly output the revised code.  Do you need to use the external knowledge base for this code review task? 
In this case, since the review suggestion is not informative enough, you can answer "Yes" and get the conversation information:
schmittjoseph: Is this error fatal?
ghost: No- the failure to initialize the virtual keyboard just means the text-to-speech buffer won't be automatically cleared after each screen. This results in a slightly degraded experience for a blind/visually impaired user, but it is still usable and is no reason to kill the installer.
schmittjoseph: Good to know. In that case I'd recommend two things:
1. Add a comment stating this is a non-fatal error.
2. Either clear err here or use a temporary err variable instead to ensure the error never gets returned by the main routine in the future if a refactor happens.
Then, you will know how to accurately revise the original code and generate:
Value:  func (ai *AttendedInstaller) initializeUI() (err error) {
 	ai.keyboard, err = speakuputils.CreateVirtualKeyboard()
 	if err != nil {
		// Non-fatal - results in a slightlydegraded experience due to the lack of a
		// text-to-speech buffer clear between views, but not bad enough to exit outright
 		logger.Log.Warnf("Failed to initialize virtual keyboard via uinput")
		err = nil
 	}
 	const osReleaseFile = "/etc/os-release"
 	ai.backdropStyle = tview.Theme{
In another case, if the review suggestion is enough for revising, for example, given the original code:
lastErr = err
 		}
 		p := bo.Pause()
		if cerr := sleep(ctx, p); cerr != nil {
 			if lastErr != nil {
				return wrappedCallErr{cerr: cerr, wrappedErr: lastErr}
 			}
			return cerr
 		}
 	}
 }
Give you the review suggestion:
nit: should this be `cErr` or `ctxErr`? or is c for call? (ln 59 says the context error)
You believe the suggestion is informative enough and you can directly revise the code following it, then when asked Do you need to use the external knowledge base for this code review task? 
You should answer No and directly output the revised code:
lastErr = err
 		}
 		p := bo.Pause()
		if ctxErr := sleep(ctx, p); ctxErr != nil {
 			if lastErr != nil {
				return wrappedCallErr{ctxErr: ctxErr, wrappedErr: lastErr}
 			}
			return ctxErr
 		}
 	}
 }
Now, give you a new original code:
            '''
            new_prompt += processed_hunk
            new_prompt += " and given the review suggestion: "
            new_prompt += comment
            messages = [
                {"role": "system",
                 "content": new_prompt + " If you feel the current comment is not informative enough for revising the code, you can choose to exploit the conversation information, otherwise you can directly output the revised code. Do you need to use the external knowledge base for this code review task? Reply Yes or No!"},
                {"role": "user", "content": "Yes or No"}
            ]
            response = completion_with_backoff(model="gpt-3.5-turbo", messages=messages, max_tokens=100)
            if "Yes" in response["choices"][0]["message"]["content"]:
                repo, pr_id = data['repo'], int(data['ghid'])
                key = (repo, pr_id)
                conversation = conversation_dict.get(key)
                if conversation:
                    # print('get')
                    new_prompt += "Now give you the context conversation information about the review suggestion: "
                    for item in conversation:
                        new_prompt += ("User: {}".format(item['user']))
                        new_prompt += ("Comment: {}".format(item['comment']))
            new_prompt += " Please output the refined code following the suggestion, note that you should only output the code without other explanations."
            if len(new_prompt) > 4096:
                new_prompt = new_prompt[:4096]
            messages = [
                {"role": "system", "content": "Suppose you are a code reviewer, please revise the original code following the revised suggestion:"},
                {"role": "user", "content": new_prompt}
            ]
            response = completion_with_backoff(model="gpt-3.5-turbo", messages=messages, max_tokens=1000)
            # print(response["choices"][0]["message"]["content"])
            if "Yes" in response["choices"][0]["message"]["content"]:
                fp.write(str(cnt) + " Yes" + '\n')
            else:
                fp.write(str(cnt) + " No" + '\n')
            try:
                fp.write(response["choices"][0]["message"]["content"] + '\n')
            except:
                continue
            #result.append(processed_hunk)
    #return result, comments

file_path = 'ref-test.jsonl'
process_hunk_key(file_path)
