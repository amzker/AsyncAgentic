import asyncio
import json
from datetime import datetime
import os
import time

from AsyncAgentic.Agents import AsyncOpenAISimpleAgent

async def get_current_time(user_id: str, chat_id: str, agent_name: str) -> str:
    print(f"get_current_time called by {agent_name} for user {user_id} and chat {chat_id}")
    return datetime.now().strftime("%H:%M:%S")

async def get_weather(city: str, user_id: str, chat_id: str, agent_name: str) -> str:
    print(f"get_weather called by {agent_name} for user {user_id} and chat {chat_id}")
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

# DEFINING STOP SIGNAL FUNCTION. I AM USING SIMPLE SLEEP HERE TO SHOW EXAMPLE.
# YOU CAN USE REDIS IN TERMS OF DISTRIBUTED SYSTEMS.
# OR HOWEVER YOU WANT IT, BASICALLY YOUR FUNCTION MUST RETURN TRUE IF YOU WANT TO STOP CHAT AT ANY POINT.
# DO NOTE: THIS FUNCTION IS NOT ACTUAL REPRESENTATION OF STOP SYSTEM YOU MAKE IN PRODUCTION.
# HERE ANY PROCESS WHICH IS TAKING MORE THAN 3 SECONDS TO COMPLETE WILL BE STOPPED. BUT IN PROD YOU CAN TRIGGER THIS MANUALLY.

async def stop_chat_signal(user_id: str, chat_id: str) -> bool:
    print(f"stop_chat_signal called for user {user_id} and chat {chat_id}")
    # you will have chat_id and user_id based listner in your system
    await asyncio.sleep(3) # so agents might be running but after 3 secounds in , it will stop at that place. 

    print(f"stop_chat_signal is returning True at {datetime.now()}")
    return True

async def hook_on_function_call_end(data):
    print(f"THIS MESSAGE IS FROM HOOK ON FUNCTION CALL END: {data}")

async def main():
    agent = AsyncOpenAISimpleAgent(
        agent_name="Test_Agent",
        agent_description="Test agent for weather and time",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        context_handling_method="simple",
        max_context_length=25000,
        max_token_per_message=4000,
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
        system_prompt="You are a helpful assistant that can check time and weather",
        manual_stop_signal_function=stop_chat_signal,
        hooks={
            'on_function_call_end': hook_on_function_call_end
        }
    )
    start_time = time.perf_counter()
    print(f"STARTING CHAT AT {datetime.now()}")
    print("\nTesting multiple tool calls... & stop chat & event hooks")
    response = await agent.send_message(
        "What's the time and weather in Tokyo and London? i am testing the concurrent execution of tools. execute both tools at same time.",
        debug_print=True
    )
    print(json.dumps(response, indent=2))

    print("\nTesting conversation with history...")
    response = await agent.send_message(
        "And what about New York, USA , France , Germanay, Florida?",
        history=response["history"]["simplified"],
        debug_print=True,
    )
    print(json.dumps(response, indent=2))
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main()) 