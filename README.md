# AsyncAgentic

Fully Asynchronous No Bloat Python Agentic Framework


## Overview

AsyncAgentic is a lightweight, production-ready, fully asynchronous Python framework for building agentic systems powered by large language models (LLMs). It is designed to prioritize simplicity, extensibility, and performance, allowing developers to focus on business logic without wrestling with complex framework abstractions. The framework supports multiple LLMs, concurrent tool execution, and provides robust features like event hooks and stop signals for production-grade control.

Key principles:
- **No Bloat**: Minimal dependencies, only what's necessary.
- **Fully Asynchronous**: No blocking calls, leveraging Python's `asyncio` for performance.
- **Simple to Use**: Intuitive API, no steep learning curve.
- **Extensible and Debuggable**: Easy to extend with custom logic and debug with detailed logs.
- **Production-Ready**: Features like user_id/chat_id, stop signals, and event hooks for seamless integration into production systems.
- **No Code Execution**: By default, no LLM-generated code execution for security. A separate `AsyncAgentic[hobbyist]` package will support code execution for non-production use cases.
- **Direct Tool Execution**: No human-in-the-middle; tool calls are executed directly, and only LLM text responses are returned to the user.

## Features
- **No Bloat**: Minimal dependencies to keep the framework lightweight.
- **Fully Asynchronous**: Built with `asyncio` for non-blocking operations.
- **Simple API**: Intuitive interface for quick integration.
- **Parallel Tool Calls**: Supports concurrent execution of multiple tool calls for efficiency.
- **Event Hooks**: Customizable hooks for monitoring and extending functionality:
  - `on_function_call_start`: Triggered when a tool call begins.
  - `on_function_call_end`: Triggered when a tool call completes.
  - `on_function_call_error`: Triggered on tool call errors.
  - `on_context_overflow`: Triggered when context limits are exceeded.
  - More hooks in development (data models pending).
- **Stop Chat System**: Attach a stop signal function to halt chat execution dynamically.
- **Context Management**: Choose between "Simple" (length-based) or "Accurate" (token-based with tiktoken) context handling.
- **Forced User and Chat IDs**: Every function and hook receives `user_id` and `chat_id` for production-grade tracking and integration (e.g., cost tracking, user-specific logic).
- **No Human-in-the-Middle**: Tool calls are executed directly without exposing internal workings to users.
- **Customizable System Prompts**: Tailor agent behavior with system prompts.
- **Flexible Tool Registry**: Register tools with JSON schemas compatible with OpenAI's format.
- **Concurrent Function Execution**: Execute multiple tool calls simultaneously when enabled.
- **Context Overflow Handling**: Configurable strategies for handling context limits, including simple dropping or summary-based dropping.
- **Stable API**: Version 1 APIs will be final with no breaking changes, ensuring long-term support (LTS).

## Installation

Install the package via uv:

```bash
uv add AsyncAgentic
```

Install the package via pip:
```bash
pip install AsyncAgentic
```


# USAGE:
## Simple Agent
```python
import os
import json
import time
import asyncio
from datetime import datetime


from AsyncAgentic.Agents import AsyncOpenAISimpleAgent

# NOTICE: FORCED DEPENDECY? CHAT_ID AND USER_ID IS COMPULSORY. 
# THIS MAY FEEL WEIRD BUT YOU WILL THANK ME LATER. WHEN YOUR PRODUCTION REQUIREMENT CHANGES. LIKE AGENT SPECIFIC COSTING OR ONLY PURCHASED AGENTS ARE ACCESIBLE ETC...
async def get_current_time(user_id: str, chat_id: str, agent_name: str) -> str:
    print(f"get_current_time called by {agent_name} for user {user_id} and chat {chat_id}")
    return datetime.now().strftime("%H:%M:%S")

async def get_weather(city: str, user_id: str, chat_id: str, agent_name: str) -> str:
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
        # model="llama3-groq-tool-use",
        api_key=os.getenv("OPENAI_API_KEY"),
        context_handling_method="simple",
        max_context_length=25000,
        max_token_per_message=4000,
        # base_url="http://localhost:11434/v1", # OLLAMA ALSO WORK , SO HERE I HAVE ADDED EXAMPLE.
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


    response = await agent.send_message(
        "What's the time and weather in Tokyo and London? i am testing the concurrent execution of tools. execute both tools at same time.",
        debug_print=True
    )
    print(json.dumps(response, indent=2))

    # TEST 2: CONVERSATION WITH HISTORY , 
    # NOTE: THIS IS JUST TO SHOW YOU GUYS HOW IT WORKS. 

    print("\nTesting conversation with history...")
    response = await agent.send_message(
        "And what about New York?",
        history=response["history"]["simplified"],
        debug_print=True
    )
    with open("response.json", "w") as f:
        json.dump(response, f, indent=2)
    end_time = time.perf_counter()
    print(f"Total time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main()) 
```
RESPONSE:
```json
{
  "stop_reason": "completed",
  "history": {
    "messages": [
      {
        "role": "user",
        "content": "What's the time and weather in Tokyo and London? i am testing the concurrent execution of tools. execute both tools at same time."
      },
      {
        "role": "assistant",
        "content": "The current time is **01:55:00**. \n\nThe weather is:\n- **Tokyo**: Sunny, 22\u00b0C\n- **London**: Sunny, 22\u00b0C"
      },
      {
        "role": "user",
        "content": "And what about New York?"
      },
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_KsLd1GublEHZCvoRpTx2nDfE",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"New York\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_KsLd1GublEHZCvoRpTx2nDfE",
        "content": "Sunny, 22\u00b0C in New York"
      },
      {
        "role": "assistant",
        "content": "The weather in New York is also **Sunny** with a temperature of **22\u00b0C**."
      }
    ],
    "simplified": [
      {
        "role": "user",
        "content": "What's the time and weather in Tokyo and London? i am testing the concurrent execution of tools. execute both tools at same time."
      },
      {
        "role": "assistant",
        "content": "The current time is **01:55:00**. \n\nThe weather is:\n- **Tokyo**: Sunny, 22\u00b0C\n- **London**: Sunny, 22\u00b0C"
      },
      {
        "role": "user",
        "content": "And what about New York?"
      },
      {
        "role": "assistant",
        "content": "The weather in New York is also **Sunny** with a temperature of **22\u00b0C**."
      }
    ]
  },
  "agent_name": "Test_Agent",
  "timestamp": "2025-05-27T01:55:05.698163",
  "output": "The weather in New York is also **Sunny** with a temperature of **22\u00b0C**.",
  "usage": {
    "completion_tokens": 20,
    "prompt_tokens": 186,
    "total_tokens": 206,
    "completion_tokens_details": {
      "accepted_prediction_tokens": 0,
      "audio_tokens": 0,
      "reasoning_tokens": 0,
      "rejected_prediction_tokens": 0
    },
    "prompt_tokens_details": {
      "audio_tokens": 0,
      "cached_tokens": 0
    }
  }
}
```
## Advance Example
here we will use stop chat and event hooks.
```python
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

```

## Configuration Options

- **agent_name**: Unique identifier for the agent.
- **model**: LLM model (e.g., `gpt-4o-mini`, `llama3-groq-tool-use`).
- **api_key**: API key for the LLM provider.
- **base_url**: Optional custom endpoint for LLM (e.g., local Ollama server).
- **context_handling_method**: `simple` (length-based) or `Accurate` (token-based with tiktoken).
- **max_context_length**: Maximum context length in tokens.
- **max_token_per_message**: Maximum tokens per message.
- **max_messages_in_context**: Maximum number of messages to retain in context.
- **dropping_strategy**: `simple` or `summary_dropping` for handling context overflow.
- **prompt_when_context_overflow**: Custom prompt for context overflow scenarios.
- **prompt_when_message_is_dropped**: Custom prompt when messages are dropped.
- **keep_function_calls_in_context**: Whether to retain function calls in context.
- **max_turns**: Maximum conversation turns to prevent infinite loops.
- **tool_registry**: List of tools with their schemas and functions.
- **execute_function_concurrently**: Enable concurrent execution of multiple tool calls.
- **manual_stop_signal_function**: Custom function to signal chat termination.
- **hooks**: Dictionary of event hooks for custom logic.

## Planned Features

- **Budget Control System**: Per-chat budget limits with hooks for exceeding budgets (pending OpenAI pricing API).
- **Retry Strategy**: Configurable retry policies for rate limits, server errors, and timeouts.
- **Streaming Response Agent**: Support for streaming responses from LLMs.
- **Multi-Model Agent**: Support for orchestrating multiple LLMs in a single agent.
- **GUI Management**: Optional GUI for managing functions and agents (under consideration, may not be implemented to avoid bloat).

## Work Left
- **Context Handling**: Improve context management with advanced strategies (e.g., summary-based dropping).
- **Image Tools**: Support for processing and generating images.
- **Error Handler**: Robust error handling for tool calls and API interactions.
- **Data Models**: Data models for event hooks and stop signals.

## Roadmap

- **Version 1.0 (LTS)**: Stable release with finalized APIs, no breaking changes.
- **Context Management Enhancements**: Advanced strategies like summary-based dropping and UI integration for context visualization.
- **Budget Control**: Implement per-chat budget limits with hooks once OpenAI pricing API is available.
- **Retry Strategy**: Add configurable retry policies for production reliability.
- **Multi-Model Support**: Enable agents to orchestrate multiple LLMs.
- **Streaming Support**: Add streaming response capabilities for real-time interactions.
- **Documentation Expansion**: Detailed guides for production use cases, hooks, and stop signals.

## Contributing

Contributions are welcome! Please submit issues or pull requests to the GitHub repository. Focus on maintaining simplicity and avoiding unnecessary dependencies.

## Notes

- Do not use this framework in production till V1.0.0 is Released. 
- The framework enforces `user_id` and `chat_id` for all functions and hooks to enable production-grade tracking (e.g., cost management, user-specific logic).
- Avoid using `AsyncAgentic[hobbyist]` in production due to security risks associated with code execution.
- For production systems, rely on direct tool execution to maintain transparency and control.
- Budget control and retry strategies are planned but not implemented due to complexity and dependency concerns.
