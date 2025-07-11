from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Any

from .base_labeler import BaseLabeler
from .schemas import IncorrectSpan, ResponseModel, FactVerificationResponse, ReasonModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .utils import find_change_spans, max_substring_match
from typing import Any

from .base_labeler import BaseLabeler

class ContextSumReasonLabeler(BaseLabeler):
    def __init__(self, model: str, prompt: str, **kwargs):
        super().__init__(model, **kwargs)
        self.llm_structured = self.llm.with_structured_output(ReasonModel, method="json_schema")

        # System/User prompt loading
        summarizer_sys_prompt = open(f"prompts/summarizer_sys_prompt.md").read()
        summarizer_user_prompt = open(f"prompts/summarizer_user_prompt.md").read()
        summarizer_prompt = ChatPromptTemplate.from_messages(
            [("system", summarizer_sys_prompt), ("user", summarizer_user_prompt)]
        )
        self.summarizer_chain = summarizer_prompt | self.llm | StrOutputParser()
        labeler_sys_prompt = open(f"prompts/{prompt}_sys.md").read()
        labeler_user_prompt = open(f"prompts/{prompt}_user.md").read()
        labeler_prompt = ChatPromptTemplate.from_messages(
            [("system", labeler_sys_prompt), ("user", labeler_user_prompt)]
        )
        self.context_dep_chain = labeler_prompt | self.llm_structured

    def label(self, question: str, answer: str, context: str, logger=None) -> list[dict[str, Any]]:
        # Summarize the context
        context_summary = self.summarizer_chain.invoke(
            {"question": question, "answer": answer, "context": context}
        )

        # Run reasoning and span detection
        response = self.context_dep_chain.invoke(
            {"question": question, "answer": answer, "context": context_summary}
        )

        if logger:
            logger.info("====================")
            logger.info(f"---Question---:\n{question}\n")
            logger.info(f"---Answer---:\n{answer}\n")
            logger.info(f"---Context---:\n{context}\n")
            logger.info(f"---Context summary---:\n{context_summary}\n")
            logger.info(f"---Reasoning Steps---:\n{response.reasoning_steps}\n")
            logger.info(f"---Response---:\n{response.model_dump_json(indent=4)}\n")

        # Generate labels with reasoning steps
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

class ContextReasonLabeler(BaseLabeler):
    def __init__(self, model: str, prompt: str, **kwargs):
        super().__init__(model, **kwargs)

        sys_prompt = open(f"prompts/{prompt}_sys.md",encoding="utf-8").read()
        user_prompt = open(f"prompts/{prompt}_user.md").read()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", sys_prompt), ("user", user_prompt)]
        )

        self.llm_structured = self.llm.with_structured_output(
            ReasonModel, method="json_schema"
        )

        self.chain = self.prompt | self.llm_structured

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, Any]]:
        response = self.chain.invoke(
            {"question": question, "answer": answer, "context": context}
        )
        if logger:
            logger.info("========================")
            logger.info(f"---Question---:\n{question}\n")
            logger.info(f"---Abswer---:\n{answer}\n")
            logger.info(f"---COntext---:\n{context}\n")
            logger.info(f"---Reasoning Steps---:\n{response.reasoning_steps}\n")
            logger.info(f"---Response---:\n{response.model_dump_json(indent=4)}\n")
        return self._get_soft_label_3(answer, response.incorrect_spans)

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
    @staticmethod
    def _get_soft_label_3(answer: str, spans: list[IncorrectSpan]) -> list[dict[str, Any]]:
        """ maximum substring match """
        result = []
        for span in spans:
            start, end = max_substring_match(answer, span.text)
            # if (end - start) >= len(span.text) // 2:  # at least half of the span is matched
            result.append({"start": start, "end": end, "prob": span.probability})
        return result