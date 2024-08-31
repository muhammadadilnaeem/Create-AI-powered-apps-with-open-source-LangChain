import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the chat model
gpt4o = ChatOpenAI(model_name='gpt-4o-mini')

# Load tools with allow_dangerous_tools set to True
tools = load_tools(
    ["llm-math", "requests_all", "human"], 
    llm=gpt4o, 
    allow_dangerous_tools=True
)

# Get the tool names
tool_names = ", ".join([tool.name for tool in tools])

# Create a prompt template with required variables
prompt_template = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=(
        "You are an AI assistant tasked with creating a simple matplotlib plot showing the sine function. "
        "Here are the tools available: {tools}. "
        "The user has asked: {input}. "
        "You can use your scratchpad to keep track of your thoughts: {agent_scratchpad}. "
        "The available tools are: {tool_names}. "
        "Please provide only the Python code necessary for this task."
    )
)

# Create the agent using the new method
agent = create_react_agent(tools=tools, llm=gpt4o, prompt=prompt_template)

# Create an AgentExecutor to manage the agent execution
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run the agent with the desired command using the invoke method
result = agent_executor.invoke(
    {"input": "Create a simple matplotlib plot showing the sine function."},
    handle_parsing_errors=True  # Enable parsing error handling
)
print(result)

