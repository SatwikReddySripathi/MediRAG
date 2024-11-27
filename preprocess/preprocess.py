import json

def preprocess_pubmedqa(input_file, output_file, long_answer=False):
    with open(input_file, "r") as f:
        data = json.load(f)
    
    documents = []
    for pmid, details in data.items():
        question = details.get("QUESTION", "")
        context = details.get("CONTEXTS", [""])[0]  # Use the first context
        answer = details.get("LONG_ANSWER" if long_answer else "final_decision", "")
        text = f"{question} {context} {answer}"
        documents.append({"text": text, "metadata": {"source": pmid}})
    
    with open(output_file, "w") as f:
        json.dump(documents, f, indent=4)

# File paths
input_file = "../data/pqaa_train_set.json"  # Update as needed
output_file = "../datasets/pqaa_documents.json"
preprocess_pubmedqa(input_file, output_file, long_answer=False)