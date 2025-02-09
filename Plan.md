So my goal is to create a fully asynchronous multiLLM agentic framework.
which is not bloated and production ready.
main goal is user have full control over system and they dont need to fight the framework and can focus on their business logic.


how i am planning it to work:

we will have lot of different types od agent such as 
streaming response agent
multimodel agent
simpleagent


```python
from AsyncAgentinc import AsyncOpenAISimpleAgent

agent = AsyncOpenAISimpleAgent(
    agent_name="My Agent",
    agent_description="this agent get date and time and return it",
    model="gpt-4o-mini",
    api_key="your_api_key",
    temperature=0.5,
    manual_stop_signal_function=None, # function which should return bool , true if chat need to be stopped and false if not, however you want to implement it , redis , websocket or whatever your stop button will do.
    check_for_stop_signal_time=3, # time is secounds to check for every x secounds for stop signal
    base_url="https://api.openai.com/v1", # just in case you want to use local llama or something which has same openai format.
    system_prompt="You are a helpful assistant",
    context_handling_method="Accurate"# or "Simple", accurate will use tiktoken and simple will use simple length and this will improved lot of performance
    max_context_length=100000, # in tokens. this is the max context length of the model
    max_token_per_message=5000, # in tokens. this is the max token per message
    max_messages_in_context=20 # amount of messages to keep in context
    dropping_strategy="simple" # or "summary_dropping" , summary dropping will drop the past messages and keep the summary of the messages in context
    prompt_when_context_overflow="Context is full, there for response is cut short , use whatever context is avalible and only responed whatever is left , do not extraplorate or assume the next context , rather explain user that query is broad and it will be helpful to do specific query"
    prompt_when_message_is_dropped="Previous messages are dropped because of context overflow"
    keep_function_calls_in_context=True # if you want to keep function calls in context otherwise it will be dropped after execution and only user and ai response will be kept
    max_turns=10 # max turn of conversation, before stopping the chat and returning the response , this is to prevent infinite loop and to save the cost
    tool_registry=[{
            name="get_current_time",
            function_schema=get_current_time_schema, # this jsonic schema of the function by openai.
            func=get_current_time, # main execution function
        },
        {
            name="get_weather",
            function_schema=get_weather_schema, # this jsonic schema of the function by openai.
            func=get_weather, # main execution function
        }
    ],
    execute_function_concurrently=True) # if enabled multiple function calls will be executed concurrently ( if agent responsed with multiple function calls) , otherwise it will be executed one by one but in both cases will be sent to openai only after all function calls are executed, if multiple is enabled then function will be return with proper context that this output is from x function call 


response = await agent.send_message("what is the weather in tokyo", history=[],debug_print=True)

return response.output # to your code , or like users. keep in mind .output will be the final response from the agent.


# structure of the response i am thinking of is

{
    "output": "the weather in tokyo is sunny",
    "stop_reason": "max_turns", # or max_function_calls or  user_stop or response  # response is when task is completed and response is given by agent.
    "history": [
        {
            "role": "user",
            "content": "what is the weather in tokyo"
        },
        {
            "role": "function",
            "content": "get_weather",
            "args": {
                "city": "tokyo"
            }
        },
        {
            "role": "assistant",
            "content": "the weather in tokyo is sunny"
        },
    ],
    "history_without_function_calls": [
        {
            "role": "user",
            "content": "what is the weather in tokyo"
        },
        {
            "role": "assistant",
            "content": "the weather in tokyo is sunny"
        },
    ],
    "usage": "openai_usage_object",
    "agent_name": "My Agent",
}
```

ahhh stopping chat signal or indicator , this was sooo much needed in my project , i had to over engineer and complicate lot of stuff to get this done
here i am thinking to help devs with it so , i will have like arg stop chat signal function , which users can assign them selves , and it will be used in agents while
executing the functions so during execution if stop signal comes we stop it right away


i am also thinking to add event hooks system
so there is no need to inherit the class and override the methods , users can just use hooks and add their own logic
which is way more flexible and long term maintainable.
also i will release very stable versions and it would be LTS system
i am not going to change api as soon as V1 is there APIs will be final
as i dont like to do breaking changes, i will add new features and make it backward compatible.
i also need to add GUI to manage functions and agents and other stuff ahhh i see the irony. alright no gui.
i mean hooks are also kind of getting out of control like
should i actually support the hooks?
i do love it a lot because before i had to do manually 


```python
# response hooks right this is very helpful
hooks = {
    "on_before_request": async_callback,  # before api call
    "on_after_request": async_callback,   # after api response
    "on_token_received": async_callback   # for streaming responses # i am not in mood of supporting streaming responses right now.
}

# function events

{
    "on_function_call_start": async_callback,    # when function execution starts
    "on_function_call_end": async_callback,      # when function execution completes
    "on_function_call_error": async_callback     # when function execution fails
}


# do i need to support context management events? i mean they are useful 
# yeah i guess i will keep it as they will be pretty helpful for frontend side on UI
# if they want to show the context in UI or something like that.

{
    "on_context_overflow": async_callback,      # when context limit is reached
    "on_message_dropped": async_callback,       # when old messages are dropped
    "on_max_turns_reached": async_callback,     # when max conversation turns hit   
}

{
    "on_manual_stop": async_callback,           # when manual stop triggered
    "on_final_response": async_callback,      # when agent gives final response
}

```
well i will need to make lot of data models for this hooks
too much work man , but i guess need to do it.

with this hooks users can do:
- custom logging
- add monitoring/analytics
- create progress indicators
- handle errors
- track costs
- debug their applications
- add business-specific functionality

and only required hooks they can attach. hm

i am thinking for chat_id and user_id system , like hooks or user specific functions
are useless if there is no chat_id or user_id. like yeah without it user will need to do kind of like extend the classes
which is what i want to avoid so everything can be neat and pretty.
if everything get chat_id and user_id as argumens then it will be super easy to integrate with any system
like they can have mongodb , and redis and stuff and user config stuff
maybe they want to have bio type stuff where ai decides user memory,
huh , so this hooks will be very helpful but if i dont make custom hooks system then 
it will be pain for people , like if x function is executed then they may want to attach stuff

ahhh got it , user_id and chat_id will solve everything as every function will also get user_id and chat_id into their 
arguments and they can do whatever they want with it. but again this would force user to add chat_id and user_id into their every function, i mean that is OK not a bad thing right like if they have it they can do much more stuff
and if they dont have it then they are probably not making production grad system anyway.
they dont need to define it in schema , they can just define it in function arguments and it will be passed to the function.by my self.

ah yes during chat init or send message i can take user_id and chat_id as arguments
this will solve too many problems. 

also human input stuff will not be added like who ask in production system to the user or even reveals internal working of system to user. it is what it is , if ai calls function it will be executed.

i also want to include budget control system like

```python
budget_control = {
    "max_budget_for_this_chat": 1, # max budget per chat in USD
    "stop_on_budget_exceeded": True, # if true then chat will be stopped when budget is exceeded
    "reserved_budget": 10 # reserved budget for system 
}
```
ahh i will also need to make hook for this budget control system where 
if budget is exceeded then they can have some kind of functions, like they may want to 
implement user specific budget or something , which is very important for production systems.
this budget control will per chat but devs can use this to calculate usage across all the chats 
as i have forced user_id and chat_id into every function. let me think
how would budget control go.

oh yes see i have single message system right now , which is awesome for devs
as like they can control budget across all , of course they will have their ledger in mongodb or something
now as soon as response goes , they will process and do other costing stuff and per chat/message /init
they can set leftover user budget into max_budget_for_this_chat 

well now i am making bloatware but is it bloatware? no , production grade system needs this 
i will not make unnecessary stuff and in readme tell them that if you are not making production grade system then you dont need this framework.

now cost caluclation at my side is going to be complicated as openai pricing per tokens changes
and it is per model , so i am not going to daily update it , if i find openai pricing api then i will use it.
to automate this stuff , otherwise we will take user input , like current_cost_per_million_tokens 
this way they will have to update and framework dont need to be updated, 
now this is good as i maybe in process of other stuff or may not be avalible also
just for pricing updates you dont want to upgrade the dependencies , that is stupid.
ah yes i will not do this time , and leave budget control for now , as soon as openai releasing pricing
endpoint i will add it.

```python
retry_strategy = {
    "rate_limit": {"max_retries": 5, "backoff": "exponential"},
    "server_error": {"max_retries": 3, "backoff": "linear"},
    "timeout": {"max_retries": 2, "backoff": "fixed"}
    "backoff_base": 2,  # for exponential backoff
    "initial_delay": 1,  # seconds
    "max_delay": 60     # maximum delay between retries
}
```
alright now i think i am overkilling it.




