"""
The LLM object every CrewAI agent uses, pointed at freeLLMAPI.

CrewAI talks to models through LiteLLM. To make LiteLLM send requests to an
OpenAI-COMPATIBLE server (your local gateway) instead of the real OpenAI, the
model name must carry the `openai/` prefix and we pass base_url + api_key. That
is the whole trick - the same three values from config.py that the original
project fed to the `openai` client.
"""
from crewai import LLM

from config import API_KEY, BASE_URL, MODEL


def build_llm(temperature: float = 0.2) -> LLM:
    # "auto" -> "openai/auto": the prefix routes LiteLLM to our base_url.
    model = MODEL if "/" in MODEL else f"openai/{MODEL}"
    return LLM(
        model=model,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
    )
