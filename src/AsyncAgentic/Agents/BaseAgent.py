import asyncio
from typing import Optional, Any, Callable, Dict, Tuple
from abc import ABC, abstractmethod

from AsyncAgentic.OpenAIClientBase.AsyncOpenAIBase import AsyncOpenAIBase

class BaseAgent(ABC):
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model: str,
        api_key: str,
        user_id: Any,
        chat_id: Any,
        manual_stop_signal_function: Optional[Callable[[Any, Any], bool]] = None,
        check_for_stop_signal_time: int = 3,
        base_url: str = "https://api.openai.com/v1",
        system_prompt: str = "You are a helpful assistant",
        context_handling_method: str = "Accurate",
        max_context_length: int = 100000,
        max_token_per_message: int = 5000,
        max_messages_in_context: int = 20,
        temperature: float = 0.7,
        hooks: Optional[Dict[str, Callable]] = None
    ):
        # Validate required fields
        if not agent_name or not isinstance(agent_name, str):
            raise ValueError("agent_name must be a non-empty string")
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string")
        if user_id is None:
            raise ValueError("user_id cannot be None")
        if chat_id is None:
            raise ValueError("chat_id cannot be None")

        self.agent_name = agent_name
        self.agent_description = agent_description
        self.model = model
        self.system_prompt = system_prompt
        self.manual_stop_signal_function = manual_stop_signal_function
        self.check_for_stop_signal_time = check_for_stop_signal_time
        
        # Context management
        self.context_handling_method = context_handling_method
        self.max_context_length = max_context_length
        self.max_token_per_message = max_token_per_message
        self.max_messages_in_context = max_messages_in_context
        
        # User identification
        self.user_id = user_id
        self.chat_id = chat_id
        
        # Initialize OpenAI client
        self.client = AsyncOpenAIBase(
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
        
        # Event hooks
        self.hooks = hooks or {}

    @abstractmethod
    async def send_message(
        self,
        message: str,
        history: Optional[list] = None,
        debug_print: bool = False
    ):
        """Each agent type must implement its own message handling"""
        pass

    async def _trigger_hook(self, hook_name: str, data: Dict[str, Any]):
        """Trigger event hook if it exists"""
        if hook_name in self.hooks:
            try:
                await self.hooks[hook_name](data)
            except Exception as e:
                # Log hook error but don't fail the main operation
                print(f"Error in hook {hook_name}: {str(e)}")

    async def _stop_function_handler(self):
        """
        Handle stop function, this will keep running until it returns true
        this is to check for stop signal in every x seconds
        """
        while True:
            if self.manual_stop_signal_function:
                try:
                    if await self.manual_stop_signal_function(
                        user_id=self.user_id, 
                        chat_id=self.chat_id
                    ):
                        return True
                except Exception as e:
                    print(f"Error in stop signal function: {str(e)}")
            await asyncio.sleep(self.check_for_stop_signal_time)
        
        return False

    async def _run_with_stop_handler(self, function: Callable, *args, **kwargs) -> Tuple[Any, bool]:
        """
        Run a function with the stop handler
        Returns:
            Tuple[Any, bool]: (result, was_stopped)
        """
        stop_task = None
        function_task = None
        try:
            stop_task = asyncio.create_task(self._stop_function_handler())
            function_task = asyncio.create_task(function(*args, **kwargs))
            
            done, pending = await asyncio.wait(
                [stop_task, function_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            if stop_task in done:
                if function_task and not function_task.done():
                    function_task.cancel()
                return None, True
            else:
                if stop_task and not stop_task.done():
                    stop_task.cancel()
                return function_task.result(), False

        finally:
            # Clean up the specific tasks for this operation
            for task in [t for t in [stop_task, function_task] if t and not t.done()]:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        
