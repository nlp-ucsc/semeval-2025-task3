from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
import warnings
from langchain_core.messages import HumanMessage
import os

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_openai")


class BaseLabeler(ABC):
    def __init__(self, model: str, **kwargs):
        if "search" in kwargs:
            self.search = kwargs["search"]
            del kwargs["search"]
        if "parse_span" in kwargs:
            self.parse_span = kwargs["parse_span"]
            del kwargs["parse_span"]

        if "/" in model:
            self.llm = ChatOpenAI(
                model_name=model,
                openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                **kwargs,
            )
        elif model.startswith("gpt"):
            self.llm = ChatOpenAI(model=model, **kwargs)
        elif model.startswith("o1") or model.startswith("o3"):
            del kwargs["temperature"]
            del kwargs["seed"]
            del kwargs["top_p"]
            self.llm = ChatOpenAI(model=model, **kwargs)
        elif model.startswith("claude"):
            del kwargs["seed"]
            self.llm = ChatAnthropic(model=model, **kwargs)
        elif model.startswith("deepseek-r1-distilled"):
            self.llm = ChatOpenAI(
                model_name="deepseek/deepseek-r1-distill-llama-70b",
                openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                **kwargs,
            )
        elif model.startswith("deepseek-r1"):
            self.llm = ChatOpenAI(
                model_name=f"deepseek/{model}",
                openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
                openai_api_base="https://openrouter.ai/api/v1",
                **kwargs,
            )
        elif model.startswith("o3-mini"):
            del kwargs["temperature"]
            del kwargs["seed"]
            del kwargs["top_p"]
            self.llm = ChatOpenAI(model=model, **kwargs)
        else:
            self.llm = ChatOllama(
                model=model, base_url="http://nlp-sv12.ucsc.edu:11434", **kwargs
            )

    def invoke_llm_with_tools(self, llm_with_tools, query: str, tools: dict) -> list:
        messages = [query]
        response = llm_with_tools.invoke(query)
        if response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                selected_tool = tools[tool_call["name"].lower()]
                tool_msg = selected_tool.invoke(tool_call)
                messages.append(tool_msg)

        return messages

    @abstractmethod
    def label(
        self, question: str, answer: str, context: str = None
    ) -> list[tuple[int, int]]:
        pass
