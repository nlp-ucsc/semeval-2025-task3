import re
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_community.chat_models.perplexity import ChatPerplexity
from typing import Any

from .base_labeler import BaseLabeler
from .schemas import IncorrectSpan, ResponseModel
from .utils import find_change_spans


class ContextFreeSimpleLabeler(BaseLabeler):
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

        self.chain = self.prompt | self.llm_structured

    def label(self, question: str, answer: str, **kwargs) -> list[dict[str, Any]]:
        response = self.chain.invoke({"question": question, "answer": answer})
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


class ContextFreeXMLLabeler(BaseLabeler):
    def __init__(self, model: str, prompt: str, **kwargs):
        assert prompt == "p5", "ContextFreeXMLLabeler only supports prompt p5"

        super().__init__(model, **kwargs)
        sys_prompt = open(f"prompts/{prompt}_sys.md").read()
        user_prompt = open(f"prompts/{prompt}_user.md").read()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", sys_prompt), ("user", user_prompt)]
        )
        self.chain = self.prompt | self.llm | self.extract_answer_and_hallu_spans

    def label(self, question: str, answer: str, **kwargs) -> list[dict[str, Any]]:
        clean_answer, hallu_spans = self.chain.invoke({"question": question, "answer": answer})
        start_idx = answer.find(clean_answer)
        if start_idx < 0:
            return []
        result = []
        for span in hallu_spans:
            result.append({"start": span[0] + start_idx, "end": span[1] + start_idx, "prob": 1.0})
        return result

    @staticmethod
    def extract_answer_and_hallu_spans(message: AIMessage) -> tuple[str, list[list[int]]]:
        # Step 1: Extract the result text wrapped in <answer></answer>
        start_result = message.content.find("<answer>") + len("<answer>")
        end_result = message.content.rfind("</answer>")
        result_text = message.content[start_result:end_result].strip()

        # Step 2: Extract a list of start and end indexes based on the <hallu> labeling
        hallu_spans = []
        offset = 0
        pattern = r"<hallu>(.*?)</hallu>"
        
        # Get all matches of hallucinated spans
        for match in re.finditer(pattern, result_text):
            # Get the text inside the hallu tags
            hallu_text = match.group(1)
            # Calculate start index accounting for previous tags
            start_idx = match.start() - offset
            # Calculate end index accounting for previous tags
            end_idx = start_idx + len(hallu_text)
            hallu_spans.append([start_idx, end_idx])
            # Update offset to account for the tags
            offset += len("<hallu></hallu>")
        
        # Step 3: Remove all hallu tags from the result text to get clean answer
        clean_answer = re.sub(pattern, r"\1", result_text)
        
        return clean_answer, hallu_spans
    

class ContextFreeMinEditLabeler(BaseLabeler):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)

        sys_prompt = dedent("""
            Correct the answer to the question.

            You will be given a question and an answer to the question. The answer may not be correct. You need to make the minimum number of changes to the answer to make it correct.

            Return the corrected answer wrapped in <corrected_answer> tags.
            
            Note: Do not correct for spelling mistakes.
        """).strip()
        user_prompt = dedent("""
            <question>
            {question}
            </question>

            <answer>
            {answer}
            </answer>
        """).strip()
        self.prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", user_prompt)])
        self.correct_answer_chain = self.prompt | self.llm | self.parse_corrected_answer
        

    def label(self, question: str, answer: str, logger=None, **kwargs) -> list[dict[str, Any]]:
        corrected_answer = self.correct_answer_chain.invoke({"question": question, "answer": answer})
        results, edit_steps = find_change_spans(answer, corrected_answer)

        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Corrected Answer ======:\n{corrected_answer}\n")
            logger.info(f"====== Edit Steps ======:")
            for step in edit_steps:
                logger.info(f" - {step}")
            logger.info(f"====== Results ======:")
            for result in results:
                logger.info(f" - {answer[result['start']:result['end']]} ({result['prob']})")
        return results

    @staticmethod
    def parse_corrected_answer(message: AIMessage) -> str:
        # breakpoint()
        start_idx = message.content.find("<corrected_answer>") + len("<corrected_answer>")
        end_idx = message.content.rfind("</corrected_answer>")
        return message.content[start_idx:end_idx].strip()


class PerplexityReasoningLabeler(BaseLabeler):
    """ No excessive prompts """

    def __init__(self, model: str, **kwargs):
        assert model == "sonar-reasoning", "PerplexityReasoningLabeler only supports sonar-reasoning"
        llm = ChatPerplexity(model="sonar-reasoning")
        user_prompt = dedent("""
            <question>
            {question}
            </question>

            <answer>
            {answer}
            </answer>
                             
            Given the question and answer, correct the answer to the question with the minimum number of changes, and return the corrected answer wrapped in <corrected_answer> tags.
            
            Note: Do not correct for spelling mistakes.
        """).strip()
        self.prompt = ChatPromptTemplate.from_messages([("user", user_prompt)])
        self.chain = self.prompt | llm | self.parse_corrected_answer

    def label(self, question: str, answer: str, context: str = None, logger=None) -> list[dict[str, Any]]:
        outputs, corrected_answer = self.chain.invoke({'question': question, 'answer': answer})
        results, edit_steps = find_change_spans(answer, corrected_answer)
        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Outputs ======:\n{outputs}\n")
            logger.info(f"====== Corrected Answer ======:\n{corrected_answer}\n")
            logger.info(f"====== Edit Steps ======:")
            for step in edit_steps:
                logger.info(f" - {step}")
            logger.info(f"====== Results ======:")
            for result in results:
                logger.info(f" - {answer[result['start']:result['end']]} ({result['prob']})")
        return results

    @staticmethod
    def parse_corrected_answer(message: AIMessage) -> str:
        start_idx = message.content.find("<corrected_answer>") + len("<corrected_answer>")
        end_idx = message.content.rfind("</corrected_answer>")
        return message.content, message.content[start_idx:end_idx].strip()
    