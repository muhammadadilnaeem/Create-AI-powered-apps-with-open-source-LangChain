

"""

Cover Letter Prompt with Template

This example shows how to use the PromptTemplate to generate a cover letter using the llm and user input.

"""

# Import Libraries
import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Set your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the model
llm = ChatOpenAI(
    model_name="gpt-4o-mini"
)

# Define a PromptTemplate to format the prompt with user input
prompt = PromptTemplate(
    input_variables=["position", "company", "skills"],
    template="Dear Hiring Manager,\n\nI am writing to apply for the {position} position at {company}. I have experience in {skills}.\n\nThank you for considering my application.\n\nSincerely,\n[Your Name]",
)

# Define a function to generate a cover letter using the llm and user input
def generate_cover_letter(position: str, company: str, skills: str) -> str:
    formatted_prompt = prompt.format(position=position, company=company, skills=skills)
    response = llm.invoke(formatted_prompt)
    return response.content  # Access the content attribute

# Define the Gradio interface inputs
inputs = [
    gr.Textbox(label="Position"),
    gr.Textbox(label="Company"),
    gr.Textbox(label="Skills")
]

# Define the Gradio interface output
output = gr.Textbox(label="Cover Letter")

# Launch the Gradio interface
gr.Interface(fn=generate_cover_letter, inputs=inputs, outputs=output).launch(server_name="0.0.0.0", server_port=7860, share=True)
