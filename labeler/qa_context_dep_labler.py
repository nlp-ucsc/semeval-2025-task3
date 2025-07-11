import re
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from .base_labeler import BaseLabeler
from .utils import find_change_spans, max_substring_match
from .schemas import ResponseModel, IncorrectSpan


class QAContextDepMinEditLabeler(BaseLabeler):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        if model.startswith("o1"):
            user_prompt = dedent("""
                <context>{context}</context>

                ---
                Please correct the original answer with minimum changes, and the corrected answer should be of similar length as the original answer. 
                Wrap the corrected answer in <corrected_answer></corrected_answer> tags.
            """).strip()
        else:
            user_prompt = dedent("""
                <context>{context}</context>
                
                ---
                Please correct the original answer with minimum changes. 
                The corrected answer should be of similar length as the original answer. Think about how the original answer can be corrected with minimum changes, and then provide the corrected answer. Wrap you thinking process in <thinking></thinking> tags. Finally, wrap the corrected answer in <corrected_answer></corrected_answer> tags.
            """).strip()
        self.prompt = ChatPromptTemplate.from_messages([
            ("user", "<question>{question}</question>"),
            ("assistant", "<answer>{answer}</answer>"),
            ("user", user_prompt),
        ])

        self.chain = self.prompt | self.llm | self.parse_thinking_and_corrected_answer

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, int | float]]:
        thinking, corrected_answer = self.chain.invoke({"question": question, "answer": answer, "context": context})
        results, edit_steps = find_change_spans(answer, corrected_answer)
        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Context from QA ======:\n{context}\n")
            logger.info(f"====== Thinking ======:\n{thinking}\n")
            logger.info(f"====== Corrected Answer ======:\n{corrected_answer}\n")
            logger.info(f"====== Edit Steps ======:")
            for step in edit_steps:
                logger.info(f" - {step}")
            logger.info("\n")
            logger.info(f"====== Results ======:")
            for result in results:
                logger.info(f" - {answer[result['start']:result['end']]} ({result['prob']})")
            logger.info("\n")
        return results

    @staticmethod
    def parse_thinking_and_corrected_answer(response: AIMessage) -> tuple[str, str]:
        """
        Parses the input string to extract the contents of <thinking> and <corrected_answer> tags.
        
        Args:
            response (AIMessage): The response ai message containing the <thinking> and <corrected_answer> tags.
            
        Returns:
            tuple: A tuple (thinking_text, corrected_answer_text).
                Returns (None, None) if neither tag is found.
        """
        # Regex patterns to match the contents of <thinking> and <corrected_answer>
        thinking_pattern = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
        corrected_pattern = re.compile(r"<corrected_answer>(.*?)</corrected_answer>", re.DOTALL)

        thinking_match = thinking_pattern.search(response.content)
        corrected_match = corrected_pattern.search(response.content)
        
        thinking_text = thinking_match.group(1).strip() if thinking_match else None
        corrected_text = corrected_match.group(1).strip() if corrected_match else None
        
        return thinking_text, corrected_text
    

class QAContextDepJsonLabeler(BaseLabeler):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        user_prompt = dedent("""
            <context>{context}</context>
            
            ---
            Based on the above context, find minimum spans of text in the original answer:
            <answer>Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China.</answer>
            where the spans of text make the answer false. In addition, give a confidence score ranging from 0 to 1 in the "probability" field, indicating how confident you are about your choice.

            Note: make your span selections as short as possible and ignore spelling mistakes.

            Answer in json format:
            ```json
            {{
            "incorrect_spans": [
                {{
                "text": "[identified incorrect span]",
                "probability": [confidence_score]
                }},
                {{
                "text": "[another identified incorrect span]",
                "probability": [confidence_score]
                }}
            ]
            }}
            ```
        """).strip()
        self.prompt = ChatPromptTemplate.from_messages([
            ("user", "<question>{question}</question>"),
            ("assistant", "<answer>{answer}</answer>"),
            ("user", user_prompt),
        ])

        self.chain = self.prompt | self.llm.with_structured_output(ResponseModel, method="json_schema")

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, int | float]]:
        response = self.chain.invoke({"question": question, "answer": answer, "context": context})
        results = self._get_soft_label_3(answer, response.incorrect_spans)
        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Context from QA ======:\n{context}\n")
            logger.info(f"====== JSON Response ======:\n{response.model_dump_json(indent=4)}\n")
            logger.info(f"====== Results ======:")
            for result in results:
                logger.info(f" - {answer[result['start']:result['end']]} ({result['prob']})")
            logger.info("\n")
        return results

    @staticmethod
    def _get_soft_label_3(answer: str, spans: list[IncorrectSpan]) -> list[dict[str, int | float]]:
        """ maximum substring match """
        result = []
        for span in spans:
            start, end = max_substring_match(answer, span.text)
            # if (end - start) >= len(span.text) // 2:  # at least half of the span is matched
            result.append({"start": start, "end": end, "prob": span.probability})
        return result
