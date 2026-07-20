import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct" #TODO: Update to 3.3-70B-Instruct if needed
FINE_TUNED_DIR = "../../llama3.1-8B-ft-clarification"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load tokenizer + model + LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_DIR)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
    device_map="auto" if DEVICE.startswith("cuda") else None,
)
model = PeftModel.from_pretrained(base_model, FINE_TUNED_DIR)
model.to(DEVICE)
model.eval()

test_data = [
    {"prompt": "My code isn't working as expected.", "label": 4},
    {"prompt": "<html>\n<head>\n    <title>My Page</title>\n</head>\n<body>\n    <h1>Welcome</h1>\n    <p>\n        <!-- TODO: Add more content -->\n", "label": 3},
    {"prompt": "I'm not sure how to implement a feature.", "label": 4},
    {"prompt": "class Animal:\n    def __init__(self, name):\n        self.name = name\n    # TODO: Create a method to make sound", "label": 3},
    {"prompt": "I can't figure out how to connect to a database.", "label": 4},
    {"prompt": "function fetchUserData(userId) {\n    // TODO: fetch user data from API\n    return userData;\n}\n\nlet data = fetchUserData(1);", "label": 3},
    {"prompt": "class Vehicle:\n    def __init__(self, make, model):\n        self.make = make\n        # TODO: Add model attribute\n", "label": 3},
    {"prompt": "#include <stdio.h>\n\nint main() {\n    int num = 10;\n    printf(\"The number is: %d\", num);\n    // TODO: Add more logic\n    return ;\n}", "label": 3},
    {"prompt": "def calculate_area(radius):\n    area = 3.14 * radius ** 2\n    return area\n\n# TODO: handle negative radius", "label": 3},
    {"prompt": "function calculateSum(arr) {\n    let sum = 0;\n    for (let i = 0; i < arr.length; i++) {\n        // TODO: Handle non‑numeric values\n    }\n    return sum;\n}", "label": 3},
    {"prompt": "I'm having trouble with my project.", "label": 4},
    {"prompt": "import numpy as np\n\ndef compute_matrix():\n    return np.array([[1, 2], [3, 4]])\n\ncompute_matrix(", "label": 3},
    {"prompt": "def connect_to_db():\n    connection = None\n    # TODO: Establish database connection\n    return connection\n", "label": 3},
    {"prompt": "def calculate_sum(a, b):\n    return a + b\n\nresult = calculate_sum(5, )", "label": 3},
    {"prompt": "I don't understand why it's not functioning.", "label": 4},
    {"prompt": "def send_email(to, subject, body):\n    import smtplib\n    server = smtplib.SMTP('smtp.example.com')\n    # TODO: implement email sending\n", "label": 3},
    {"prompt": "I need help with my user interface.", "label": 4},
    {"prompt": "def read_file(file_path):\n    with open(file_path, 'r') as file:\n        # TODO: Read and process file content\n", "label": 3},
    {"prompt": "var myObject = {\n    name: 'ChatGPT',\n    // TODO: Add properties\n};\n\nconsole.log(myObject);", "label": 3},
    {"prompt": "function fetchData(url) {\n    // TODO: Implement fetch logic\n    return;\n}\n\nfetchData('http://example.com')\n", "label": 3},
    {"prompt": "const data = [1, 2, 3];\nconst result = data.map(num => num * 2;\nconsole.log(result);  // TODO: handle empty array", "label": 3},
    {"prompt": "I have an issue with my API call.", "label": 4},
    {"prompt": "I'm confused about a concept in programming.", "label": 4},
    {"prompt": "public class Example {\n    private int value;\n\n    public Example(int value) {\n        this.value = value;\n    }\n}\n\n// TODO: Add getters", "label": 3},
    {"prompt": "const numbers = [1, 2, 3, 4];\nconst doubled = numbers.map(num => num * 2);\nconsole.log(doub);  // Typo in variable name", "label": 3},
    {"prompt": "var myArray = [];\n\nfunction addElement(element) {\n    myArray.push(element);\n    // TODO: handle duplicates\n}", "label": 3},
    {"prompt": "public class Sample {\n    private int number;\n    \n    // TODO: Add constructor\n}\n\nSample s = new Sample();", "label": 3},
    {"prompt": "There's an issue with my API integration.", "label": 4},
    {"prompt": "I'm confused about data structures.", "label": 4},
    {"prompt": "I need help with a function.", "label": 4},
    {"prompt": "I'm stuck on a part of my code.", "label": 4},
    {"prompt": "Can someone explain how to optimize my code?", "label": 4},
    {"prompt": "function sendRequest(url) {\n    const xhr = new XMLHttpRequest();\n    xhr.open('GET', url);\n    // TODO: Add onload handler\n}", "label": 3},
    {"prompt": "import numpy as np\n\ndef compute_mean(data):\n    return np.mean(data)\n\n# Missing data check\nprint(compute_mean())", "label": 3},
    {"prompt": "I need guidance on a library.", "label": 4},
    {"prompt": "I'm having trouble with my function.", "label": 4},
    {"prompt": "def calculate_area(radius):\n    return 3.14 * radius ** 2\n\n# TODO: Handle negative radius\narea = calculate_area(-5)", "label": 3},
    {"prompt": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\n# TODO: Add a view function\n\ndef main():\n    return 'Hello World'", "label": 3},
    {"prompt": "class User:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n\nuser = User('Alice', )  # TODO: specify age", "label": 3},
    {"prompt": "while True:\n    print('Running...')\n    if not condition:\n        break  // TODO: Define condition", "label": 3},
    {"prompt": "I need assistance with a function.", "label": 4},
    {"prompt": "import requests\n\n# TODO: Ensure headers are set for API request\nresponse = requests.get('https://api.example.com/data')\nprint(response)", "label": 3},
    {"prompt": "I'm facing issues with deploying my website.", "label": 4},
    {"prompt": "I'm confused about this library.", "label": 4},
    {"prompt": "I need assistance with debugging.", "label": 4},
    {"prompt": "import pandas as pd\n\ndf = pd.read_csv('data.csv')\n\n# TODO: Clean data\nprint(df.head())", "label": 3},
    {"prompt": "<html>\n<head>\n    <title>My Page</title>\n</head>\n<body>\n    <h1>Welcome to my page\n    <p>This is a paragraph.", "label": 3},
    {"prompt": "I'm not sure how to structure my project.", "label": 4},
    {"prompt": "function fetchData(url) {\n    // TODO: handle errors\n    let response = await fetch(url);\n    return response.json();\n}", "label": 3},
    {"prompt": "My program isn't working as expected.", "label": 4},
    {"prompt": "import numpy as np\n\ndata = np.array([1, 2, 3])\nprint(dat)\n# Missing function to process data", "label": 3},
    {"prompt": "def calculate_area(radius):\n    return math.pi * radius ** 2\n\n# TODO: Handle negative radius values", "label": 3},
    {"prompt": "I want to improve my code's performance.", "label": 4},
    {"prompt": "I can't figure out how to integrate something.", "label": 4},
]

def generate_reasoning(prompt: str, label: int, max_new_tokens: int = 50):
    instr = (
        f"<start_of_turn>user\n"
        f"{prompt}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    tokens = tokenizer(instr, return_tensors="pt", padding=True)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

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

    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return full_text.split("<start_of_turn>model\n")[-1].strip()

# 5. Run and save
if __name__ == "__main__":
    results = []
    for ex in test_data:
        reasoning = generate_reasoning(ex["prompt"], ex["label"])
        print("─" * 40)
        print(f"Prompt:  {ex['prompt']}")
        print(f"Label:   {ex['label']}")
        print(f"Reason:  {reasoning}\n")
        results.append({
            "prompt": ex["prompt"],
            #"label": ex["label"],
            "response": reasoning
        })

    with open("results_finetune1.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} entries to results_finetune1.json")