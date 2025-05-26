from typing import Callable, Optional, Any, List, Dict
import asyncio
from datetime import datetime
import json
from termcolor import colored
import tiktoken
from AsyncAgentic.Agents.BaseAgent import BaseAgent

class AsyncOpenAISimpleAgent(BaseAgent):
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model: str,
        api_key: str,
        tool_registry: Optional[List[Dict[str, Any]]] = None,
        execute_function_concurrently: bool = True,
        debug_print: bool = False,
        context_handling_method: str = "simple", # accurate or simple
        max_context_length: int = 100000,
        max_token_per_message: int = 5000,
        max_messages_in_context: int = 20,
        prompt_when_context_overflow: str = "Context is full, there for response is cut short , use whatever context is avalible and only responed whatever is left , do not extraplorate or assume the next context , rather explain user that query is broad and it will be helpful to do specific query",
        prompt_when_message_is_dropped: str = "Previous messages are dropped because of context overflow",
        manual_stop_signal_function : Optional[Callable[[Any, Any], bool]] = None,
        check_for_stop_signal_time: int = 3,
        hooks: Optional[Dict[str, Callable]] = None,
        **kwargs
    ):
        super().__init__(
            agent_name=agent_name,
            agent_description=agent_description,
            model=model,
            api_key=api_key,
            context_handling_method=context_handling_method,
            max_context_length=max_context_length,
            max_token_per_message=max_token_per_message,
            max_messages_in_context=max_messages_in_context,
            manual_stop_signal_function=manual_stop_signal_function,
            check_for_stop_signal_time=check_for_stop_signal_time,
            hooks=hooks,
            **kwargs
        )
        self.tool_registry = tool_registry or []
        self.execute_function_concurrently = execute_function_concurrently
        self._tool_map = {tool["name"]: tool for tool in self.tool_registry}
        self._message_history = []  # Track complete conversation
        self.debug_print = debug_print
        self.prompt_when_context_overflow = prompt_when_context_overflow
        self.prompt_when_message_is_dropped = prompt_when_message_is_dropped

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to max_tokens based on context handling method"""
        if self.context_handling_method.lower() == "accurate":
            try:
                # TODO: I SHOULD PROABALY SWITCH TO THREADS HERE. OR KEEP SIMPLE CONTEXT HANDLING ONLY.
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("gpt-4o")
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            print(f"Truncating text from {len(tokens)} to {max_tokens} tokens")
            truncated = encoding.decode(tokens[:max_tokens])
            return truncated + "\n[Context trimmed due to token limit]"
        else:
            # Simple method: 1 token = 4 characters
            if len(text) <= max_tokens * 4:
                return text
            print(f"Truncating text from {len(text)} to {max_tokens * 4} characters")
            return text[:max_tokens * 4] + "\n[Context trimmed due to token limit]"

    def _get_total_context_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Get total token count for all messages"""
        if self.context_handling_method.lower() == "accurate":
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("gpt-4")
            return sum(len(encoding.encode(msg["content"])) for msg in messages if msg.get("content"))
        else:
            return sum(len(msg["content"]) // 4 for msg in messages if msg.get("content"))

    def _trim_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        trimmed_messages = []
        was_trimmed = False
        for msg in messages:
            if msg.get("content"):
                trimmed_content = self._truncate_tokens(msg["content"], self.max_token_per_message)
                if trimmed_content != msg["content"]:
                    was_trimmed = True
                trimmed_messages.append({**msg, "content": trimmed_content})
            else:
                trimmed_messages.append(msg)

        messages_dropped = False
        while self._get_total_context_tokens(trimmed_messages) > self.max_context_length:
            if len(trimmed_messages) <= 1: 
                break
            if self.debug_print:
                print(colored("Dropping oldest message due to context overflow", 'red'))
            trimmed_messages.pop(1)  # keep system message, drop oldest user/assistant message
            messages_dropped = True

        # If any trimming happened, add context overflow prompt to the last message
        if was_trimmed or messages_dropped:
            last_msg = trimmed_messages[-1]
            overflow_note = "\n\n" + self.prompt_when_context_overflow
            if messages_dropped:
                overflow_note = "\n\n" + self.prompt_when_message_is_dropped + overflow_note
            trimmed_messages[-1] = {**last_msg, "content": last_msg["content"] + overflow_note}

        return trimmed_messages

    def _prepare_messages(self, message: str, history: Optional[list] = None) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API with context handling"""
        messages = []
        
        if not history or history[0]["role"] != "system":
            messages.append({"role": "system", "content": self.system_prompt})
        
        if history:
            messages.extend(history)
        
        if not history or history[-1]["content"] != message:
            messages.append({"role": "user", "content": message})
        
        # Apply context handling
        messages = self._trim_context(messages)
        
        if self.debug_print:
            total_tokens = self._get_total_context_tokens(messages)
            print(colored(f"Total context tokens: {total_tokens}/{self.max_context_length}", 'cyan'))
        
        return messages

    async def send_message(
        self,
        message: str,
        history: Optional[list] = None,
        debug_print: bool = False
    ):
        try:
            safe_history = self._validate_history(history)
            self._message_history = safe_history
            
            self._add_to_history({
                "role": "user",
                "content": message
            })

            await self._trigger_hook("on_before_request", {
                "message": message,
                "user_id": self.user_id,
                "chat_id": self.chat_id,
                "timestamp": datetime.now().isoformat()
            })

            messages = self._prepare_messages(message, self._message_history)
            tools = [{"type": "function", "function": tool["function_schema"]} 
                    for tool in self.tool_registry]

            while True:  # continue until we get a response without tool calls
                if debug_print:
                    print(colored(f"Sending messages to OpenAI: {messages}", 'cyan'))

                result, was_stopped = await self._run_with_stop_handler(
                    self.client.send_message,
                    messages=messages,
                    model=self.model,
                    tools=tools if tools else None,
                    user_id=self.user_id,
                    chat_id=self.chat_id
                )

                if was_stopped:
                    await self._trigger_hook("on_manual_stop", {
                        "user_id": self.user_id,
                        "chat_id": self.chat_id,
                        "timestamp": datetime.now().isoformat()
                    })
                    return self._format_response(result, stop_reason="manual_stop")

                response = result
                # if debug_print:
                #     print(colored(f"Response: {response}", 'green'))

                assistant_message = {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }
                if response.choices[0].message.tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in response.choices[0].message.tool_calls
                    ]
                self._add_to_history(assistant_message)
                messages.append(assistant_message)

                # if no tool calls, we're done
                if not response.choices[0].message.tool_calls:
                    break

                tool_calls = response.choices[0].message.tool_calls
                
                if self.execute_function_concurrently:
                    result, was_stopped = await self._run_with_stop_handler(
                        self._execute_tools_concurrent,
                        tool_calls
                    )
                else:
                    result, was_stopped = await self._run_with_stop_handler(
                        self._execute_tools_sequential,
                        tool_calls
                    )

                if was_stopped:
                    return self._format_response(response, stop_reason="manual_stop")

                tool_results = result

                for result in tool_results:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": str(result["result"])
                    }
                    self._add_to_history(tool_message)
                    messages.append(tool_message)

                if debug_print:
                    for tool_call in tool_calls:
                        print(colored(f"Executing {tool_call.function.name} with arguments: {tool_call.function.arguments}", 'yellow'))

            await self._trigger_hook("on_after_request", {
                "response": response,
                "user_id": self.user_id,
                "chat_id": self.chat_id,
                "timestamp": datetime.now().isoformat()
            })

            return self._format_response(response, stop_reason="completed")

        except Exception as e:
            await self._trigger_hook("on_error", {
                "error": str(e),
                "user_id": self.user_id,
                "chat_id": self.chat_id,
                "timestamp": datetime.now().isoformat()
            })
            raise e

    async def _execute_tools_concurrent(self, tool_calls):
        """execute multiple tool calls concurrently"""
        tasks = []
        for tool_call in tool_calls:
            tool = self._tool_map.get(tool_call.function.name)
            if tool:
                tasks.append(self._execute_tool(tool, tool_call))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if isinstance(r, Exception):
                raise r
        
        return results

    async def _execute_tools_sequential(self, tool_calls):
        """execute tool calls one at a time"""
        results = []
        for tool_call in tool_calls:
            tool = self._tool_map.get(tool_call.function.name)
            if tool:
                result = await self._execute_tool(tool, tool_call)
                if result:
                    results.append(result)
        return results

    async def _execute_tool(self, tool, tool_call):
        """execute a single tool"""
        await self._trigger_hook("on_function_call_start", {
            "function": tool_call.function.name,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "agent_name": self.agent_name
        })

        try:
            if isinstance(tool_call.function.arguments, str):
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = tool_call.function.arguments

            args = {
                **arguments,
                "user_id": self.user_id,
                "chat_id": self.chat_id,
                "agent_name": self.agent_name
            }
            
            result = await tool["func"](**args)

            await self._trigger_hook("on_function_call_end", {
                "function": tool_call.function.name,
                "result": result,
                "user_id": self.user_id,
                "chat_id": self.chat_id,
                "agent_name": self.agent_name
            })

            return {
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "result": result
            }

        except Exception as e:
            await self._trigger_hook("on_function_call_error", {
                "function": tool_call.function.name,
                "error": str(e),
                "user_id": self.user_id,
                "chat_id": self.chat_id,
                "agent_name": self.agent_name
            })
            # raise e
            return {
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "result": str(e)
            }

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format tool results for OpenAI API"""
        messages = []
        
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": results[0]["tool_call_id"],  # all results are from same assistant message
                "type": "function",
                "function": {
                    "name": result["name"],
                    "arguments": result.get("arguments", "{}")
                }
            } for result in results]    
        })
        
        for result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": str(result["result"])
            })
        
        if self.debug_print:
            print(colored(f"Formatted messages for OpenAI: {messages}", 'magenta'))
        return messages

    def _format_response(self, response: Any, stop_reason: str) -> Dict[str, Any]:
        """Format the final response with complete history"""
        complete_history = self._get_complete_history()
        
        clean_history = []
        for msg in complete_history:
            if msg["role"] == "user":
                clean_history.append(msg)
            elif msg["role"] == "assistant":
                if msg.get("content"):
                    clean_history.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })

        result = {
            "stop_reason": stop_reason,
            "history": {
                "messages": complete_history,
                "simplified": clean_history
            },
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat()
        }

        if response is None:
            result.update({
                "output": None,
                "usage": None
            })
        else:
            result.update({
                "output": response.choices[0].message.content,
                "usage": response.usage.model_dump()
            })

        return result

    def _add_to_history(self, message: Dict[str, Any]):
        """add message to history"""
        self._message_history.append(message)

    def _get_complete_history(self) -> List[Dict[str, Any]]:
        """get complete conversation history"""
        return self._message_history.copy()

    def _validate_history(self, history: Optional[list]) -> list:
        """validate and return safe history copy"""
        if not history:
            return []
        
        if not isinstance(history, list):
            raise ValueError("History must be a list")
        
        for msg in history:
            if not isinstance(msg, dict):
                raise ValueError("History messages must be dictionaries")
            if "role" not in msg:
                raise ValueError("History messages must have 'role' field")
            if "content" not in msg:
                raise ValueError("History messages must have 'content' field")
            
        return history.copy()
