# Import necessary functions and modules from Utils.py
from Utils import get_vectordb, data_loader, get_llm, get_embeddings, process_llm_response

# Import relevant modules
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import gradio as gr  # Import Gradio for creating a chat interface

# Define a list of URLs to load data from
urls = ["https://www.infobellit.com/index.html", "https://www.infobellit.com/products.html", "https://www.infobellit.com/services.html", "https://www.infobellit.com/aboutus.html"]

# Define a template for prompting questions
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible. say "Ask if you have any other queries".
Strictly avoid repeating the answers.

{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


# Load data from the specified URLs
data = data_loader(urls)

# Get embeddings based on the "BGE" model
embeddings = get_embeddings(model="BGE")

# Create a Together Large Language Model (LLM) instance
llm = get_llm(temp=0.1, max_tokens=512)

# Create a retriever using vector data and embeddings
retriever = get_vectordb(data, embeddings).as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create a chain for question answering using the LLM and retriever
QA_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, return_source_documents=True)


# Define a function for the chatbot
def chatbot(message, history):
    # Use the QA chain to generate a response
    llm_response = QA_chain(message)
    response = process_llm_response(llm_response)
    return response


# Create a Gradio chat interface for the chatbot
demo = gr.ChatInterface(chatbot, title="Infobell QA_Chatbot")

# Launch the Gradio interface and share it
demo.launch(share=True)
