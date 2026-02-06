import httpx
from openai import OpenAI
from openai import AzureOpenAI
import tenacity
import os


OPENAI_MODEL_NAME_GPT4 = "gpt-4-turbo"  # new gpt-4-turbo
OPENAI_MODEL_NAME_GPT35 = "gpt-3.5-turbo"
OPENAI_MODEL_NAME_GPT4o = "gpt-4o"
OPENAI_MODEL_NAME_MAP = {
    "openai-gpt-4": OPENAI_MODEL_NAME_GPT4,
    "openai-gpt-35": OPENAI_MODEL_NAME_GPT35,
    "openai-gpt-4o": OPENAI_MODEL_NAME_GPT4o,
}

'''
openai_client = OpenAI(
    http_client=httpx.Client(
        limits=httpx.Limits(
            max_connections=1000,
            max_keepalive_connections=100
        )
    )
)
'''

openai_client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(http2=True, verify=False)
)

@tenacity.retry(#wait=tenacity.wait_random_exponential(min=60, max=600),
                stop=tenacity.stop_after_attempt(2), 
                #reraise=True
               )
def api_call_single(client: OpenAI, model: str, messages: list[dict], temperature: float = 0.0, thinking=False, **kwargs):
    # Call the API
    if not thinking:
        messages[0]['content'] = '/no_think '+messages[0]['content']
    print('api_call_single', client.base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # Ensure messages is a list
        temperature=temperature,
        **kwargs
    )
    return response

@tenacity.retry(#wait=tenacity.wait_random_exponential(min=60, max=600),
                stop=tenacity.stop_after_attempt(2), 
                #reraise=True
               )
def api_function_call_single(client: OpenAI, model: str, messages: list[dict], tools: list[dict], temperature: float = 0.0,thinking=False, **kwargs):
    # Call the API
    if not thinking:
        messages[0]['content'] = '/no_think '+messages[0]['content']
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        temperature=temperature,
        **kwargs
    )
    return response

def call_openai(llm: str, messages: list[dict], temperature: float = 0.0,thinking=False, **kwargs):
    """
    Call the OpenAI API asynchronously to a list of messages using high-level asyncio APIs.
    """
    print('call_openai', os.getenv("BASE_URL"))
    model = llm#OPENAI_MODEL_NAME_MAP.get(llm)
    if model is None:
        raise ValueError(f"Unsupported LLM model: {llm}")
    response = api_call_single(openai_client, model, messages, temperature,thinking, **kwargs)
    return response