# This will be the base class for Async simple openai client

from typing import Optional, Dict, Any, List
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

class AsyncOpenAIBase:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
    ):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.temperature = temperature

    async def send_message(
        self,
        messages: list[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        user_id: Any = None,
        chat_id: Any = None,
    ) -> ChatCompletion:
        """
        Base method to send message to OpenAI API
        """
        try:
            if tools:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=self.temperature
                )
            else:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature
                )
            return response
        except Exception as e:
            # Base error handling - agents will implement their retry logic
            raise e
