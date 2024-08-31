from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import gradio as gr
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the prompt template for the conversation
_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""

# Create the prompt from the template
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)

# Set up memory components
conv_memory = ConversationBufferMemory(memory_key="history")

# Initialize the ConversationChain with the prompt and memory
conversation_chain = ConversationChain(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", api_key=openai_api_key),
    prompt=PROMPT,
    memory=conv_memory,
)

# Set up the Gradio interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()  # No need to specify type as 'tuples'
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        # Append the new message to the chat history
        chat_history.append((message, ""))
        
        # Prepare the history for the model
        history = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in chat_history])
        
        # Invoke the conversation chain with current input and history
        bot_response = conversation_chain.invoke({"input": message, "history": history})
        
        # Print the response to inspect its structure
        print(bot_response)  # For debugging purposes
        
        # Access the correct response from the bot
        bot_message = bot_response.get("response", "Sorry, I didn't understand that.")  # Ensure 'response' is the correct key
        
        # Update the last entry with the bot's response
        chat_history[-1] = (message, bot_message)  # Use a tuple format
        time.sleep(1)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)