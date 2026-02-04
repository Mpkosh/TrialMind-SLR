import httpx
from openai import OpenAI
from openai import AzureOpenAI
import tenacity
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
#BASE_URL = "https://router.huggingface.co/v1"
BASE_URL= "https://gigachat.devices.sberbank.ru/api/v1"
#HUGGINGFACE_HUB_TOKEN ='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'
HUGGINGFACE_HUB_TOKEN='eyJjdHkiOiJqd3QiLCJlbmMiOiJBMjU2Q0JDLUhTNTEyIiwiYWxnIjoiUlNBLU9BRVAtMjU2In0.YRJxY_Gpcr0OTl66zliP9vq2lCwdsOVUk2oxOfE_FwBI-teifDVHdPHP5OiU2lk4r8jwA9KDTSnRLFYPRc7zTey5F2068fN_nA4wHwY2BC5YMENX-0RyFpsdKAQDtt0bAk_NuYu89n5MMCrFfTMW5Gfx73Y8-T3RSguM9jcNZUJIZIAlhhPD7v8OVjIhh8gL9AD1F94VvYHmdv75DQC63-JYcnqUyZx25fWd1A3SNyN6ebHqceSKsvRtCZ-TyKCdoxJDmyVC0p8GIAZfYirlXY7Rs8Be3sp_nbWiz7LlfvHv1yI6zDRJ_9K9n0aWm7NajeEPmF0QYVHmOXS6pWNdZQ.kAV9Ys5xdV2uUzzDxHyZSQ.fr70dO8KNVeVTKsnPkBRadNVB13OA5PT5ss3jr2baLZgkF2S60WYNPG-VU5A42XrVdzrmZIaok6iOGDo_PMoKw8UQ-IzAjuHehaZaS_3pODnUq0POeRLDzWBaHS6iUk0TIOEOrZZGtgbNfHwOYqbQ6kXKRaKnT7wLyMa3l-tZ8yWFK-dHB-Ocr9bhk5qliO5z7kSbOupOETwHgdMXbda_nDpK5RELaEEaYwDjzdRuGooTQwP9BcqMs4EUHbE2WFQZWp0TarL5RJx7EgD6oWBiyknRP7X3vpvnOj6XofyQ4Wl-JAFraOKqbk9cvezmb6Y6PDu7MXQs8OXsjHXBQy_Rhe3rzKx9DWZS-GgGE-vD2VG6N7WoVM1mgzJnmXLff9Bw4WjjeKGDTmnmA3-cMUFRzi5R4wmVRQ8mA473-JCAG1sqHzyEiaizJnfTifAZOHLkMUZe-3DFKsMXFxmCCtU40va5K0Nx_dxTgR1nmv2ZxPGi-U_rdSwRKA2qQwKU1ErFId9WI2tSq19pqpXh5R4tCkruewR7uEeG9FLsiHm_585gTvWLT_H-cV-Bw-miJ8gwJRUnaqoQ_9T_yrIKSLvlFe8oMOH6qJJ5eDS9IUfdu2tFffVo8GlD8XY01-qw8IOwS5N4IMda1wpz7cpaPf8m789vcNVCDWnDMISXjWROdAC9yUF1oGFlnat-RxiflqXkuLMKM2H6Sx2n51yxLSaP9HknE0QSjuZjO0oWdEhjuo.Ew6MEu9pgPk5Skl1Q708YXu4qsfAEC8mYQlT86rc2GI'

openai_client = OpenAI(
    base_url=BASE_URL,
    api_key=HUGGINGFACE_HUB_TOKEN,
    http_client=httpx.Client(http2=True, verify=False)
)


@tenacity.retry(#wait=tenacity.wait_random_exponential(min=60, max=600),
                stop=tenacity.stop_after_attempt(1), 
                #reraise=True
               )
def api_call_single(client: OpenAI, model: str, messages: list[dict], temperature: float = 0.0, thinking=False, **kwargs):
    # Call the API
    if not thinking:
        messages[0]['content'] = '/no_think '+messages[0]['content']
    print(client.base_url, client.api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # Ensure messages is a list
        temperature=temperature,
        **kwargs
    )
    return response

@tenacity.retry(#wait=tenacity.wait_random_exponential(min=60, max=600),
                stop=tenacity.stop_after_attempt(1), 
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
    model = llm#OPENAI_MODEL_NAME_MAP.get(llm)
    if model is None:
        raise ValueError(f"Unsupported LLM model: {llm}")
    response = api_call_single(openai_client, model, messages, temperature,thinking, **kwargs)
    return response