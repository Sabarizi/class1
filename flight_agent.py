# flight_agent_loop_manual.py
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

# Gemini-style OpenAI wrapper
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(model=model)

# Define the flight agent
flight_agent = Agent(
    name="Flight Booking Agent",
    instructions=(
        "You are a flight booking assistant. Ask user for:\n"
        "- Departure city\n"
        "- Destination city\n"
        "- Travel date\n"
        "Then show 3 mock flight options (AirBlue, PIA, SereneAir)\n"
        "Ask which one they want to book.\n"
        "When they select a flight, confirm with: 'Your flight has been booked successfully!'\n"
        "Until then, keep the conversation going step-by-step."
    )

)

# Start interactive loop
print("✈️ Flight Booking Assistant is active!\n")
chat_history = []

while True:
    user_input = input("🧑 You: ")

    # Append history to maintain context
    chat_history.append(f"User: {user_input}")
    full_prompt = "\n".join(chat_history)

    # Send to agent
    response = Runner.run_sync(
        flight_agent,
        input=full_prompt,
        run_config=config
    )

    assistant_output = response.final_output
    print(f"🤖 Agent: {assistant_output}")

    # Save response to history for next prompt
    chat_history.append(f"Agent: {assistant_output}")

    # Stop if booking is confirmed
    if "Your flight has been booked successfully!" in assistant_output:
        print("🎉 Booking Complete. Have a safe flight!")
        break
