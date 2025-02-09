import asyncio
import json
from datetime import datetime
import os

from AsyncAgentic.Agents import AsyncOpenAISimpleAgent
# hahaha officially this is working fine


async def get_current_time(user_id: str, chat_id: str, agent_name: str) -> str:
    """Get the current time"""
    print(f"get_current_time called by {agent_name} for user {user_id} and chat {chat_id}")
    return datetime.now().strftime("%H:%M:%S")

async def get_weather(city: str, user_id: str, chat_id: str, agent_name: str) -> str:
    """Simulate getting weather"""
    print(f"get_weather called by {agent_name} for user {user_id} and chat {chat_id}")
    await asyncio.sleep(1)
    return f"Sunny, 22°C in {city}"

get_time_schema = {
    "name": "get_current_time",
    "description": "Get the current time",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

get_weather_schema = {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city to get weather for"
            }
        },
        "required": ["city"]
    }
}

async def main():
    agent = AsyncOpenAISimpleAgent(
        agent_name="Test_Agent",
        agent_description="Test agent for weather and time",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        user_id="test_user",
        chat_id="test_chat",
        tool_registry=[
            {
                "name": "get_current_time",
                "function_schema": get_time_schema,
                "func": get_current_time
            },
            {
                "name": "get_weather",
                "function_schema": get_weather_schema,
                "func": get_weather
            }
        ],
        execute_function_concurrently=True,
        system_prompt="You are a helpful assistant that can check time and weather"
    )

    print("\nTesting single tool call...")
    response = await agent.send_message(
        "What time is it?",
        debug_print=True
    )
    print(json.dumps(response, indent=2))

    print("\nTesting multiple tool calls...")
    response = await agent.send_message(
        "What's the time and weather in Tokyo and London? i am testing the concurrent execution of tools. execute both tools at same time.",
        debug_print=True
    )
    print(json.dumps(response, indent=2))

    print("\nTesting conversation with history...")
    response = await agent.send_message(
        "And what about New York?",
        history=response["history"]["simplified"],
        debug_print=True
    )
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 