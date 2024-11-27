import json
import os

def process_file(input_file, output_file, long_answer=False):
    print(f"Processing file: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    with open(input_file, "r") as f:
        data = json.load(f)
    
    processed = []
    for pmid, details in data.items():
        question = details.get("QUESTION", "")
        context = details.get("CONTEXTS", [""])[0]  # Use the first context if multiple
        answer = details.get("LONG_ANSWER" if long_answer else "final_decision", "")
        processed.append({"question": question, "context": context, "answer": answer})
    
    with open(output_file, "w") as f:
        json.dump(processed, f, indent=4)

# Relative paths
train_file = "../data/pqaa_train_set.json"
dev_file = "../data/pqaa_dev_set.json"
pqal_train_file = "../data/pqal_fold0/train_set.json"
pqal_dev_file = "../data/pqal_fold0/dev_set.json"

# Ensure output directory exists
os.makedirs("../datasets", exist_ok=True)

# Process files
process_file(train_file, "../datasets/pqaa_train_processed.json", long_answer=False)
process_file(dev_file, "../datasets/pqaa_dev_processed.json", long_answer=False)
process_file(pqal_train_file, "../datasets/pqal_train_processed.json", long_answer=True)
process_file(pqal_dev_file, "../datasets/pqal_dev_processed.json", long_answer=True)
