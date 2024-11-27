from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json

# Load preprocessed data
with open("datasets/pqaa_documents.json", "r") as f:
    documents = json.load(f)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vectorstore
texts = [doc["text"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]
vectorstore = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)

# Save FAISS index
vectorstore.save_local("db/faiss")