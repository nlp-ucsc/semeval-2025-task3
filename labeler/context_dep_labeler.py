from typing import Any
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

from .base_labeler import BaseLabeler
from .schemas import IncorrectSpan, ResponseModel, ReasonModel
from tools import get_tools
from .utils import find_change_spans, max_substring_match


class ContextDepSimpleLabeler(BaseLabeler):
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

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, Any]]:
        while True:
            try:
                response = self.chain.invoke(
                    {"question": question, "answer": answer, "context": context}
                )
                break
            except Exception as e:
                print(f"Error getting response, error: {e}\nRetrying...")

        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Context ======:\n{context}\n")
            logger.info(f"====== Response ======:\n{response.model_dump_json(indent=4)}\n")

        if self.parse_span == "string_match" or self.parse_span is None:
            results = self._get_soft_label(answer, response.incorrect_spans)
        elif self.parse_span == "string_match_no_order":
            results = self._get_soft_label_2(answer, response.incorrect_spans)
        elif self.parse_span == "max_substring":
            results = self._get_soft_label_3(answer, response.incorrect_spans)
        else:
            raise ValueError(f"Invalid parse_span: {self.parse_span}")
        return results

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
    def _get_soft_label_2(answer: str, spans: list[IncorrectSpan]) -> list[dict[str, Any]]:
        result = []
        for span in spans:
            start = answer.find(span.text)
            if start < 0:
                continue
            end = start + len(span.text)
            result.append({"start": start, "end": end, "prob": span.probability})
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


class ContextSummarizingLabeler(BaseLabeler):
    def __init__(self, model: str, prompt: str, **kwargs):
        super().__init__(model, **kwargs)
        self.llm_structured = self.llm.with_structured_output(ResponseModel, method="json_schema")

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

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, Any]]:
        context_summary = self.summarizer_chain.invoke(
            {"question": question, "answer": answer, "context": context}
        )
        response = self.context_dep_chain.invoke(
            {"question": question, "answer": answer, "context": context_summary}
        )
        if logger:
            logger.info("====================")
            logger.info(f"---Question---:\n{question}\n")
            logger.info(f"---Answer---:\n{answer}\n")
            logger.info(f"---Context---:\n{context}\n")
            logger.info(f"---Context summary---:\n{context_summary}\n")
            logger.info(f"---Response---:\n{response.model_dump_json(indent=4)}\n")
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


class ContextAtomicFactLabeler(BaseLabeler):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        sys_prompt_extract_facts = """Identify and list individual facts within a given piece of text.

# Steps

1. **Read the Text Carefully**: Review the provided text with attention to detail.
2. **Identify Statements**: Look for individual claims, assertions, or data-driven statements that can be considered facts.
3. **Break Down Complex Sentences**: For sentences containing multiple assertions or claims, separate them into distinct factual statements.
4. **List Facts**: Clearly list each identified fact in sequential order.

# Output Format

- Provide a list of individual facts as distinct bullet points. Each fact should be a concise sentence or phrase extracted from the text.

# Examples

**Input:**
"The Great Wall of China is one of the New7Wonders of the World and was primarily built to protect the Chinese states and empires against invasions. It stretches over 13,000 miles and consists of walls, trenches, and natural barriers."

**Output:**
- The Great Wall of China is one of the New7Wonders of the World.
- The Great Wall of China was primarily built to protect the Chinese states and empires against invasions.
- The Great Wall of China stretches over 13,000 miles.
- The Great Wall of China consists of walls, trenches, and natural barriers. 

# Notes

- Focus on identifying facts and not opinions or interpretations. 
- Ensure clarity for each extracted fact to aid subsequent fact-checking steps.
- Be consistent in separating and formatting the facts for readability."""

        user_prompt_extract_facts = """Break the following piece of text down to individual facts:
'''{text}'''"""

        sys_prompt_check_facts = """Verify a list of facts against a provided context to determine their accuracy.

To perform this task, you will need to carefully compare each fact with the available context, taking into account any detailed evidence or references provided. Provide your reasoning process for each fact. If the fact directly aligns with or is supported by the context, mark it as "True." If the fact contradicts the context, lacks supporting evidence, or cannot be corroborated, mark it as "False. Provide your confidence of the labeling as a score between 0 and 1 for each fact."

# Steps

1. **Understand the Context**: Carefully read through the provided context to comprehend the information and points made.
2. **Reason**: Reason through each fact carefully:
   - Compare the fact to the information in the context.
   - Use logic and critical thinking to assess the likelihood of the fact being true based solely on the provided context.
3. **Categorize the Fact**:
   - Label the fact as "True" if it is supported by the context.
   - Label the fact as "False" if it lacks support or is explicitly contradicted by the context.
   - Provide a confidence score for each fact between 0 and 1.
# Notes

- Reason through the fact against the context before making your judgement.
- If a fact is partially supported, provide context in the explanation for this, but mark it as "True" only if the supported part is significant, and your confidence score should reflect this.
- Consider any nuances or implicit information within the context that might affect the verification.
- For facts that do not pertain directly to the provided context, use your best judgement."""

        user_prompt_check_facts = """# Context:
{context}

# Facts:
{facts}"""

        sys_prompt_find_spans = """Find the spans of text associated with false facts.

You will be given a document and a list of facts extracted from it, each labeled as true or false according to a context and reasoning. It also includes a confidence score for each fact.

# Steps

1. **Input Analysis**: Review the provided document and list of extracted facts.
2. **Find False Facts**: Identify the facts that are labeled as **False**.
3. **Fact Matching**: Match each false fact back to its corresponding span of text within the original document.
4. **Output**: Output the spans of text and confidence score as probability associated with each false fact in json format.

# Output Format

The output should be in JSON format as shown below:

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

# Notes

- Return the text spans in the order of the original document."""

        user_prompt_find_spans = """# Document:
{document}

# Facts and Verifications:
{facts}"""

        self.llm_structured = self.llm.with_structured_output(
            ResponseModel, method="json_schema"
        )

        fact_extraction_prompt = ChatPromptTemplate.from_messages(
            [("system", sys_prompt_extract_facts), ("user", user_prompt_extract_facts)]
        )
        self.fact_extraction_chain = fact_extraction_prompt | self.llm | StrOutputParser()

        fact_verification_prompt = ChatPromptTemplate.from_messages(
            [("system", sys_prompt_check_facts), ("user", user_prompt_check_facts)]
        )
        self.fact_verification_chain = fact_verification_prompt | self.llm | StrOutputParser()

        find_spans_prompt = ChatPromptTemplate.from_messages(
            [("system", sys_prompt_find_spans), ("user", user_prompt_find_spans)]
        )
        self.find_spans_chain = find_spans_prompt | self.llm_structured

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, Any]]:
        facts = self.fact_extraction_chain.invoke({'text': answer})
        # print(facts)
        verifications = self.fact_verification_chain.invoke({'context': context, 'facts': facts})
        # print(verifications)
        response = self.find_spans_chain.invoke({'document': context, 'facts': verifications})
        # print(response)
        if logger:
            logger.info("========================")
            logger.info(f"---Question---:\n{question}\n")
            logger.info(f"---Answer---:\n{answer}\n")
            logger.info(f"---Context---:\n{context}\n")
            logger.info(f"---Fact Extraction---:\n{facts}\n")
            logger.info(f"---Fact Verification---:\n{verifications}\n")
            logger.info(f"---Find Spans---:\n{response.model_dump_json(indent=4)}\n")
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


class CreateContextLabeler(BaseLabeler):
    def __init__(self, model: str, prompt: str, **kwargs):
        super().__init__(model, **kwargs)

        sys_prompt = open(f"prompts/{prompt}_sys.md").read()
        user_prompt = open(f"prompts/{prompt}_user.md").read()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", sys_prompt), ("user", user_prompt)]
        )
        self.search_tool = get_tools()[self.search]
        self.llm_structured = self.llm.with_structured_output(
            ResponseModel, method="json_schema"
        )
        self.chain = self.prompt | self.llm_structured

    def label(self, question: str, answer: str, context: str = None, logger = None) -> list[dict[str, str | float]]:
        query = f"{question}\n\nAnswer the above question in a way that can fact check the following answer:\n{answer}"
        context = self.search_tool.invoke(query)
        response = self.chain.invoke({'question': question, 'answer': answer, 'context': context})
        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Retrieved Context ======:\n{context}\n")
            logger.info(f"====== Response ======:\n{response.model_dump_json(indent=4)}\n")
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


class ContextDepMinEditLabeler(BaseLabeler):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)

        sys_prompt = dedent("""
            Correct a given answer with minimal changes based on the context and question provided within XML tags.

            # Steps

            1. **Understand the Context**: Analyze the information provided within the `<context>` tags to comprehend its details and nuances.
            2. **Analyze the Question**: Determine the specific query within the `<question>` tags and understand how it relates to the context.
            3. **Evaluate the Initial Answer**: Compare the answer inside the `<answer>` tags to the insights gained from the context. Identify any discrepancies or areas lacking precision.
            4. **Make Minimal Corrections**: Modify the answer to align correctly with the context and adequately address the question while making the least number of changes necessary.

            # Output Format

            - Wrap the corrected answer in `<corrected_answer>` tags as a short paragraph or sentence, directly addressing any errors or omissions found.

            # Example

            <context>The Earth revolves around the Sun in an elliptical orbit taking approximately 365 days.</context>

            <question>What orbits around the Sun?</question>

            <answer>The Sun orbits the Earth.</answer>

            <corrected_answer>The Earth orbits around the Sun.</corrected_answer>
        """).strip()

        user_prompt = dedent("""
            <context>{context}</context>
            <question>{question}</question>
            <answer>{answer}</answer>
        """).strip()
        self.prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("user", user_prompt)])
        self.chain = self.prompt | self.llm | self.parse_corrected_answer

    def label(self, question: str, answer: str, context: str, logger=None) -> list[dict[str, Any]]:
        corrected_answer = self.chain.invoke({'context': context, 'question': question, 'answer': answer})
        results, edit_steps = find_change_spans(answer, corrected_answer)
        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Context ======:\n{context}\n")
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
        start_idx = message.content.find("<corrected_answer>") + len("<corrected_answer>")
        end_idx = message.content.rfind("</corrected_answer>")
        return message.content[start_idx:end_idx].strip()


class ContextDepMinEditLabeler2(BaseLabeler):
    """ No excessive prompts """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)

        user_prompt = dedent("""
            Use the given context, correct the answer to the question with the minimum number of changes.

            You will be given a context, a question and an answer to the question. The answer may not be correct. You need to make the minimum number of changes to the answer to make it correct.

            Return the corrected answer wrapped in <corrected_answer> tags.
            
            Note: Do not correct for spelling mistakes.

            <context>
            {context}
            </context>

            <question>
            {question}
            </question>

            <answer>
            {answer}
            </answer>
        """).strip()
        self.prompt = ChatPromptTemplate.from_messages([("user", user_prompt)])
        self.chain = self.prompt | self.llm | self.parse_corrected_answer

    def label(self, question: str, answer: str, context: str, logger=None) -> list[dict[str, Any]]:
        corrected_answer = self.chain.invoke({'context': context, 'question': question, 'answer': answer})
        results, edit_steps = find_change_spans(answer, corrected_answer)
        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Context ======:\n{context}\n")
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
        start_idx = message.content.find("<corrected_answer>") + len("<corrected_answer>")
        end_idx = message.content.rfind("</corrected_answer>")
        return message.content[start_idx:end_idx].strip()


class ContextReasonLabeler(BaseLabeler):
    def __init__(self, model: str, prompt: str, **kwargs):
        assert prompt in ["p6", "p7"], "Invalid prompt"
        super().__init__(model, **kwargs)

        sys_prompt = open(f"prompts/{prompt}_sys.md").read()
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
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Context ======:\n{context}\n")
            logger.info(f"====== Reasoning Steps ======:\n{response.reasoning_steps}\n")
            logger.info(f"====== Response ======:\n{response.model_dump_json(indent=4)}\n")

        if self.parse_span == "string_match" or self.parse_span is None:
            results = self._get_soft_label(answer, response.incorrect_spans)
        elif self.parse_span == "string_match_no_order":
            results = self._get_soft_label_2(answer, response.incorrect_spans)
        elif self.parse_span == "max_substring":
            results = self._get_soft_label_3(answer, response.incorrect_spans)
        else:
            raise ValueError(f"Invalid parse_span: {self.parse_span}")
        return results

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
    def _get_soft_label_2(answer: str, spans: list[IncorrectSpan]) -> list[dict[str, Any]]:
        result = []
        for span in spans:
            start = answer.find(span.text)
            if start < 0:
                continue
            end = start + len(span.text)
            result.append({"start": start, "end": end, "prob": span.probability})
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
