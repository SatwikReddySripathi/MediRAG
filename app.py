import gradio as gr
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS  # Use Chroma if needed
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the vectorstore
vectorstore = FAISS.load_local("db/faiss")  # Change to Chroma if you're using ChromaDB
retriever = vectorstore.as_retriever()

# Load the LLM
model_name = "EleutherAI/gpt-neo-2.7B"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Wrap LLM with LangChain
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Create the RAG pipeline
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Define the Gradio interface
def answer_question(question):
    response = rag_chain.run(question)
    sources = response["source_documents"]
    source_texts = "\n".join([doc.metadata["source"] for doc in sources])
    return response["answer"], source_texts

interface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs=["text", "text"],
    title="RAG-Powered Medical QA System",
    description="Ask a medical question and receive a RAG-generated answer along with source references."
)

if __name__ == "__main__":
    interface.launch()
