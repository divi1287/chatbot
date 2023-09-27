# Import necessary libraries and modules
from langchain.vectorstores import Chroma  # Import Chroma for vector storage
from langchain.document_loaders import AsyncChromiumLoader  # Import loader for web page retrieval
from langchain.document_transformers import BeautifulSoupTransformer  # Import transformer for HTML parsing
from langchain.embeddings import HuggingFaceInstructEmbeddings  # Import embeddings for instruction-based models
from llm import TogetherLLM  # Import Together large language model
from langchain.embeddings import HuggingFaceBgeEmbeddings  # Import embeddings for BGE model
import textwrap  # Import textwrap for text formatting


# Function to load data from URLs
def data_loader(urls):
    # Load HTML asynchronously
    loader = AsyncChromiumLoader(urls=urls)
    html = loader.load()

    # Transform HTML using BeautifulSoup
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p", "div"])
    return docs_transformed


# Specify the model name for embeddings
model_name = "BAAI/bge-base-en"


# Function to get embeddings based on the specified model
def get_embeddings(model: str):
    if model == "instruct":
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"})
        return embeddings
    elif model == "BGE":
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings


# Directory to persist vector data
persist_directory = 'db1'


# Function to create a Vectordb from documents and embeddings
def get_vectordb(data, embeddings):
    vectordb = Chroma.from_documents(documents=data, embedding=embeddings, persist_directory=persist_directory)
    return vectordb


# Function to create a Together Large Language Model (LLM)
def get_llm(temp, max_tokens):
    llm = TogetherLLM(
        model="togethercomputer/llama-2-70b-chat",
        temperature=temp,
        max_tokens=max_tokens
    )
    return llm


# Function to wrap text while preserving newlines
def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text


# Function to process LLM response and wrap the generated text
def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response['result'])
