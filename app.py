# pip install openai-agents
# pip install python-dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
# print(gemini_api_key)
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
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
    name='Translater Agent',
    instructions=(
        "You are a Translator agent. Translate the user's message from **Roman Urdu** to English, and English to Roman Urdu.\n\n"
        "Roman Urdu means Urdu written in English letters. Example: 'mera naam saba hai'."
    )
)

# Taking input from user
user_input = input("Enter your message to Translate into English or Roman Urdu: ")

response = Runner.run_sync(
    writer,
    input=user_input,
    run_config=config
)

print("Translated Output:", response.final_output)
