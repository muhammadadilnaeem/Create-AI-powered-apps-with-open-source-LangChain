"""
Chain Customer Support

This example shows how to use the LLMChain to handle customer complaints.

"""


# Import Libraries
import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Set your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the model
llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # You might want to set your API key here if needed
    temperature=0.9  # Set the temperature to make the output more random
)

def handle_complaint(complaint: str) -> str:
    # Define the prompt template for the complaint handling
    prompt = PromptTemplate(
        input_variables=["complaint"],
        template="I am a customer service representative. I received the following complaint: {complaint}. My response is:"
    )

    # Create a language model chain with the defined prompt template
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain with the user's complaint and return the response
    return chain.run(complaint)

# Create a Gradio interface for the handle_complaint function
iface = gr.Interface(fn=handle_complaint, inputs="text", outputs="text")

# Launch the Gradio interface
iface.launch(share=True)
