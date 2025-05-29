import streamlit as st
import nest_asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Fix async loop error in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check API key
if not gemini_api_key:
    st.error("🚨 GEMINI_API_KEY is not set in your .env file.")
    st.stop()

# Set page config
st.set_page_config(page_title="Roman Urdu Translator", layout="wide")

# Theme colors
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stChatMessage.user {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        color: #000000;
    }
    .stChatMessage.assistant {
        background-color: #E6E6FA;
        padding: 10px;
        border-radius: 10px;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Setup Gemini model
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

# Translator agent
translator_agent = Agent(
    name='Roman Urdu Translator',
    instructions="You are a smart translator. Translate Roman Urdu to English. Roman Urdu is Urdu written in English alphabets."
)

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("🌐 Roman Urdu to English Translator (Gemini AI)")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"**{msg['content']}**")

# User input
user_input = st.chat_input("💬 Type your Roman Urdu sentence...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**{user_input}**")

    # Run agent
    response = Runner.run_sync(translator_agent, input=user_input, run_config=config)
    
    # Show AI response
    with st.chat_message("assistant"):
        st.markdown(f"**{response.final_output}**")

    st.session_state.messages.append({"role": "assistant", "content": response.final_output})
    st.rerun()

