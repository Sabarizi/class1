from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:  
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")       
# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",       
    openai_client=external_client
)
config = RunConfig(
    model=model,    
    tracing_disabled=True
)
# Write Agent
writer = Agent(
    name='Currency Converter Agent',
    instructions=(
        "You are a Currency Converter agent. Convert the user's message from one currency to another.\n\n"
        "Example: 'Convert 100 USD to EUR'."
    )
)
# Taking input from user
user_input = input("Enter your message to Convert Currency: ")
response = Runner.run_sync(
    writer,
    input=user_input,
    run_config=config
)
print("Converted Currency Output:", response.final_output)
# Note: Ensure that the GEMINI_API_KEY is set in your .env file before running this script.
# This script uses the OpenAI Agents library to create a currency converter agent that can convert amounts between different currencies based on user input.
# The agent is designed to understand and process currency conversion requests, providing accurate and relevant responses based on the input provided by the user.