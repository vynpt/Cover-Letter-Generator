# pip install transformers langchain tiktoken chromadb pypdf InstructorEmbedding accelerate bitsandbytes sentence-transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Pass the directory path where the model is stored on your system
model_name = "./google/flan-t5-large"

# Initialize a tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize a model for sequence-to-sequence tasks using the specified pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a pipeline for text-to-text generation using a specified model and tokenizer
pipe = pipeline(
    "text2text-generation",  # Specify the task as text-to-text generation
    model=model,              # Use the previously initialized model
    tokenizer=tokenizer,      # Use the previously initialized tokenizer
    max_length=512,           # Set the maximum length for generated text to 512 tokens
    temperature=0,            # Set the temperature parameter for controlling randomness (0 means deterministic)
    top_p=0.95,               # Set the top_p parameter for controlling the nucleus sampling (higher values make output more focused)
    repetition_penalty=1.15   # Set the repetition_penalty to control the likelihood of repeated words or phrases
)

# Create a Hugging Face pipeline for local language model (LLM) using the 'pipe' pipeline
local_llm = HuggingFacePipeline(pipeline=pipe)

# Generate text by providing an input prompt to the LLM pipeline and print the result
print(local_llm('translate English to Vietnamese: How old are you?'))