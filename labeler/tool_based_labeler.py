from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any
from tools import get_tools

from .base_labeler import BaseLabeler
from .schemas import IncorrectSpan, ResponseModel, FactVerificationResponse


class ToolDepSimpleLabeler(BaseLabeler):
    def __init__(self, model: str, prompt: str, **kwargs):
        super().__init__(model, **kwargs)

        sys_prompt = open(f"prompts/{prompt}_sys.md").read()
        user_prompt = open(f"prompts/{prompt}_user.md").read()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", sys_prompt), ("user", user_prompt)]
        )

        self.llm_structured = self.llm.with_structured_output(
            ResponseModel, method="json_schema"
        )
        tool_names = ["search_you","search_gpt"]
        self.tool_dict = get_tools()
        tools = [self.tool_dict[name] for name in tool_names]
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.chain = self.prompt | self.llm_with_tools

    def label(self, question: str, answer: str, context: str,logger=None) -> list[dict[str, Any]]:
        response_messages = self.invoke_llm_with_tools(self.llm_with_tools,question,self.tool_dict)
        response = self.llm_structured.invoke(response_messages)
        return self._get_soft_label(answer, response.incorrect_spans)

    @staticmethod
    def _get_soft_label(
        answer: str, spans: list[IncorrectSpan]
    ) -> list[dict[str, Any]]:
        result = []
        curr_idx = 0
        for span in spans:
            start = answer.find(span.text, curr_idx)
            if start < 0:
                continue
            end = start + len(span.text)
            result.append({"start": start, "end": end, "prob": span.probability})
            curr_idx = end
        return result
