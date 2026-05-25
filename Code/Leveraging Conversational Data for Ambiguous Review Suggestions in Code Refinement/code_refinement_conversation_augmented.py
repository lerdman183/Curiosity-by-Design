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



bootstrap = {
    "role": "system",
    "content": "You are an expert code reviewer. You should follow the example and answer the last question. you will now refine the given code following the reviewed suggestion.",
}
use_tool = {
    "role": "user",
    "content": '''Note that if you think the current review suggestion is not informative or confused to refine the code, you can choose to exploit the external knowledge, i.e., the conversation between the code submiter and the reviewer, or do not use the tool if you believe the current review suggestion is informative enough for refine the original code. "
               Please first answer "Yes, exploit the external knowledge" or "No, do not exploit the external knowledge." and give the reason. Note that you should exploit the external knowledge when it is really necessary i.e., the review suggestion is really not clear enough.'''
}
external_knowledge = {
    "role": "user",
    "content": "Give the conversation about the review comment."
}
final_get_answer = {
    "role": "user",
    "content": "Base on the context, please output the refined code following the reviewed suggestion or the external knowledge. Note that you should output a refined version even if you think the code needs not to be refined."
}
prompt_1_chat = [
    {
        "role": "user",
        "content": '''given the original code: 
func (ai *AttendedInstaller) initializeUI() (err error) {
 	ai.keyboard, err = speakuputils.CreateVirtualKeyboard()
 	if err != nil {
 		logger.Log.Warnf("Failed to initialize virtual keyboard via uinput")
 	}
 	const osReleaseFile = "/etc/os-release"
}
Given the review suggestion:
Is this error fatal? 
Please output the revised code following the review suggestion.
    '''
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Yes, exploit the external knowledge about the conversation around the review suggestion. Because in this case, the review suggestion is not informative enough because it does not point out which error is and how to revise the original code."
    },
    {
        "role": "user",
        "content": ''' Here is the conversation!
schmittjoseph: Is this error fatal?
ghost: No. the failure to initialize the virtual keyboard just means the text-to-speech buffer won't be automatically cleared after each screen. This results in a slightly degraded experience for a blind/visually impaired user, but it is still usable and is no reason to kill the installer.
schmittjoseph: Good to know. In that case I'd recommend two things:
1. Add a comment stating this is a non-fatal error.
2. Either clear err here or use a temporary err variable instead to ensure the error never gets returned by the main routine in the future if a refactor happens.
        '''
    },
    final_get_answer,
    {
        "role": "assistant",
        "content": '''Now I know how to accurately revise the original code and output:
func (ai *AttendedInstaller) initializeUI() (err error) {
 	ai.keyboard, err = speakuputils.CreateVirtualKeyboard()
 	if err != nil {
		// Non-fatal - results in a slightlydegraded experience due to the lack of a
		// text-to-speech buffer clear between views, but not bad enough to exit outright
 		logger.Log.Warnf("Failed to initialize virtual keyboard via uinput")
		err = nil
 	}
 	const osReleaseFile = "/etc/os-release"
}      
                   '''
    }
]
prompt_2_chat = [
    {
        "role": "user",
        "content": '''given the original code:
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
Given the review suggestion:
nit: should this be `cErr` or `ctxErr`? or is c for call? (ln 59 says the context error)
Please output the revised code following the review suggestion.
    '''},
    use_tool,
    {
        "role": "assistant",
        "content": "No, do not exploit the external knowledge because in this case, the review suggestion is informative enough because it requires to change `cErr` to `ctxErr`. I have captured this without the need of external knowledge."
    },
    final_get_answer,
    {
        "role": "assistant",
        "content": '''Now, I can directly revise the original code and output:
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
                   '''
    }
]
start_prompt = None
def call_chat_completion(messages, num_retri, stop_word='Problem: '):
    global retrieval_demos, start_prompt
    del_prompt = num_retri
    len_prompt = [len(prompt_1_chat), len(prompt_2_chat)]
    while (True):
        try:
            res = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_word
            )
            break
        except openai.error.InvalidRequestError as e:    # Note here #############
            if (del_prompt < 2):
                messages = start_prompt + messages[len(start_prompt) + len_prompt[del_prompt]:]
                del_prompt += 1
            else:
                if (len(messages) < 5):
                    retrieval_demos = retrieval_demos.split('Problem: ')[:-1]
                    if (len(retrieval_demos) == 0):
                        messages = start_prompt + messages[7:]
                    else:
                        retrieval_demos = [demo.strip() for demo in retrieval_demos]
                        retrieval_demos = '\nProblem: '.join(retrieval_demos)
                        retrieval_demos = retrieval_demos.strip()
                        start_prompt[3] = {"role": "user",
                                           "content": begin_prompt["user_retrieval"].format(retrieval_demos)}
                        messages = start_prompt + messages[len(start_prompt):]
                else:
                    messages = start_prompt + messages[7:]

        except openai.error.RateLimitError as e:
            time.sleep(10)
        except:
            time.sleep(5)

    choice = res['choices'][0]
    steps = choice['message']['content'].strip()
    if "Problem: " in steps:
        steps = steps.split('Problem: ')[0].strip()
    if "Q: " in steps:
        steps = steps.split('Q: ')[0].strip()
    return steps


begin_prompt = {
    "user_begin": "You should solve the problem step by step and you should follow the react in the history. In each reasoning step, you can use external knowledge (conversation about the review suggestion) to help you solve problem. Do you understand?",
    "assistant_begin": "Yes, I understand. I will follow my response in the conversation history and solve the problem step by step.",
    "user_retrieval": "I give you some similar problems.\n{}You can use the knowledge and thoery in these problem. Do you understand?",
    "assistant_retrieval": "Yes, I understand. I will solve the problem step by step and use tool to help me.",
    "user_intr_tool": "You can use external knowledge to help you solve the problem and I give you the instruction of tools usage. External knowledge can help you get the conversation about the review suggestion if you think it is not informative enough. Do you understand?",
    "assistant_intr_tool": "Yes, I understand. I will choose to use the external knowledge if needed to help me solve the problem.",
}
def process_hunk_key(file_path):
    #result = []
    #comments = []
    cnt = 0
    with open(file_path, 'r') as file, open('ans_code_refinement_chatcot_yesno.txt', 'a') as fp, open('message_conversation_yesno.txt', 'a') as fpp:
        for line in file:

            data = json.loads(line)
            hunk_content = data['hunk']
            comment = data['comment']
            #comments.append(comment)
            processed_hunk = "\n".join(process_hunk_line(line) for line in hunk_content.split("\n"))
            global start_prompt
            start_prompt = [
                bootstrap,
                {"role": "user", "content": begin_prompt["user_intr_tool"]},
                {"role": "assistant", "content": begin_prompt["assistant_intr_tool"]},
                #{"role": "user", "content": begin_prompt["user_retrieval"].format(retrieval_demos)},
                #{"role": "assistant", "content": begin_prompt["assistant_retrieval"]},
            ]
            messages = start_prompt
            annotated_examplars = [prompt_1_chat, prompt_2_chat]
           # for i in range(0, 2):
            messages = messages + annotated_examplars[0] + annotated_examplars[1]
            problem = '''Give you the original code: 
            '''
            problem += processed_hunk
            problem += " and given the review suggestion: "
            problem += comment
            messages = messages + [
                {"role": "user", "content": begin_prompt["user_begin"]},
                {"role": "assistant", "content": begin_prompt["assistant_begin"]},
                {"role": "user",
                 "content": problem + " Please revise the code following the review suggestion. Let's think step by step and choose whether to use the external knowledge to solve the code refinement task."},
            ]
            # give more examples that need the conversation information
            messages = messages + [use_tool]
            response =  call_chat_completion(messages, 0)
            messages += [{"role": "assistant", "content": response}]
            print_messages = [{"role": "assistant", "content": response}]
            if "yes, exploit the external knowledge" in response.lower():
                fp.write(str(cnt) + " Yes" + '\n')
            else:
                fp.write(str(cnt) + " No" + '\n')
            if "yes, exploit the external knowledge" in response.lower():
                repo, pr_id = data['repo'], int(data['ghid'])
                key = (repo, pr_id)
                conversation = conversation_dict.get(key)
                if conversation:
                    # print('get')
                    new_prompt = "Here is the conversation data about the review suggestion: "
                    for item in conversation:
                        new_prompt += ("User: {}".format(item['user']))
                        new_prompt += ("Comment: {}".format(item['comment']))
                    messages += [{"role": "user", "content": new_prompt}]
                    print_messages += [{"role": "user", "content": new_prompt}]
            messages = messages + [final_get_answer]
            response = call_chat_completion(messages, 0)
            dialogue_string = "\n".join([item['content'] for item in print_messages])
            try:
                fp.write(response + '\n')
                fpp.write(dialogue_string + '\n')
            except:
                continue
            #result.append(processed_hunk)
    #return result, comments

file_path = 'ref-test.jsonl'
process_hunk_key(file_path)
