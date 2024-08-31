"""
Document Summarizer

This example shows how to use the VectorstoreIndexCreator to create an index and to summarize and query documents.

"""


# Import Libraries
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings, OpenAI  # Updated import path for OpenAIEmbeddings and OpenAI
import wget
import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Link to a text document
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0W2REN/images/state_of_the_union.txt"
output_path = "state_of_the_union.txt"  # Local path for the file

# Check if the file already exists
if not os.path.exists(output_path):
    # Download the file using wget
    wget.download(url, out=output_path)

# Load the document with specified encoding
loader = TextLoader(output_path, encoding='utf-8')

# Create an embedding model instance using the updated class
embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")

# Create the index instance that makes it searchable, using the specified embedding model
index = VectorstoreIndexCreator(embedding=embedding_model).from_loaders([loader])

# Create an OpenAI instance for querying
llm = OpenAI(api_key=openai_api_key, temperature=0)

# Function to summarize the document
def summarize(query):
    return index.query(query, llm=llm)

# Running into Gradio
iface = gr.Interface(fn=summarize, inputs="text", outputs="text")
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
