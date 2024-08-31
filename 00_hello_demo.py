import os
from langchain_openai import ChatOpenAI
import gradio as gr
from dotenv import load_dotenv
from langchain.schema import HumanMessage

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI model
llm = ChatOpenAI(
    model_name="gpt-4o-mini"
)

# Define the chatbot function using the invoke method
def chatbot(input_text):
    response = llm.invoke([HumanMessage(content=input_text)])
    return response.content  # Access the content of the AIMessage directly

# Set up the Gradio interface with appropriate input and output components
demo = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="GPT-4o-mini Chatbot", description="Chat with gpt-4o-mini!")

# Launch the Gradio app on the specified server and port
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
