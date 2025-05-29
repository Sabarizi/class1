import streamlit as st
import nest_asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

nest_asyncio.apply()

# Load .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY not found in your .env file.")
    st.stop()

# Page settings
st.set_page_config(page_title="Ask Anything GPT", layout="wide")

# Custom styling
st.markdown("""
<style>
.stChatMessage.user {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    color: black;
}
.stChatMessage.assistant {
    background-color: #E6E6FA;
    padding: 10px;
    border-radius: 10px;
    color: black;
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

# GENERAL Purpose Chat Agent
chat_agent = Agent(
    name='AskAnythingGPT',
    instructions="You are a helpful assistant that answers any kind of question. Be friendly, clear, and informative like ChatGPT."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.title("🧠 Ask Anything GPT (Gemini Powered)")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"**{msg['content']}**")

# User input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run Gemini agent
    response = Runner.run_sync(chat_agent, input=user_input, run_config=config)

    # Show response
    with st.chat_message("assistant"):
        st.markdown(response.final_output)

    st.session_state.messages.append({"role": "assistant", "content": response.final_output})
    st.rerun()
