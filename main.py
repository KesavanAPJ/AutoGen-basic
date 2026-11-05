
import asyncio
import os

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Load Gemini API Key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API Key not found in .env file")


# Connect Gemini LLM with AutoGen
model_client = OpenAIChatCompletionClient(
    model="gemini-3.5-flash",
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",    
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
        "structured_output": False
    }
)


# Create a simple AI Agent
agent = AssistantAgent(
    name="my_first_agent",
    model_client=model_client,
    system_message="""
    You are a helpful AI assistant.
    Answer user questions in simple English.
    Keep your answers short and clear.
    """
)


# Run the Agent
async def main():

    print("================================")
    print("   MY FIRST AUTOGEN AI AGENT")
    print("================================")

    question = input("\nAsk your question: ")

    response = await agent.run(task=question)

    print("\nAI Agent Response:")
    print(response.messages[-1].content)

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
