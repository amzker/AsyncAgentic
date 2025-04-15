# here i am planning to create separate types of context handlers
import asyncio
import tiktoken

def get_accurate_token_count(message: str, model: str) -> int:
    try:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError as e:
            encoding = tiktoken.get_encoding("gpt-4o")
    except Exception as e:
        return len(message) / 4
    return len(encoding.encode(message))


def get_simple_token_count(message: str) -> int:
    return len(message) / 4


async def get_token_count_from_history(history: list, model: str, method: str) -> int:
    if method.lower() == "accurate":
        return sum(await asyncio.to_thread(get_accurate_token_count, message, model) for message in history)
    elif method.lower() == "simple":
        return sum(get_simple_token_count(message) for message in history)
    else:
        raise ValueError(f"Invalid method: {method}")
