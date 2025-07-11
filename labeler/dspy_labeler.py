import dspy

from .base_labeler import BaseLabeler
from .utils import max_substring_match


class FindIncorrectSpans(dspy.Signature):
    '''Given a context, a question and an answer, find incorrect spans of text in the answer'''

    context: str = dspy.InputField(desc='context of the question')
    question: str = dspy.InputField(desc='question')
    answer: str = dspy.InputField(desc='answer to the question')
    incorrect_spans: list[str] = dspy.OutputField(desc='incorrect spans of text in the answer, must be substrings of the answer')
    confidences: list[float] = dspy.OutputField(desc='confidences about each labeled incorrect span, between 0 and 1')


class DSPySimpleLabeler(BaseLabeler):
    def __init__(self, model: str, cot: bool = False, **kwargs):
        temperature = kwargs.get('temperature', 0.0)
        max_tokens = kwargs.get('max_tokens', 8192)
        if model.startswith("gpt") or model.startswith("o1"):
            model_str = f"openai/{model}"
        else:
            model_str = f"ollama_chat/{model}"
        self.parse_span = kwargs.get('parse_span', None)
        
        lm = dspy.LM(model_str, temperature=temperature, max_tokens=max_tokens, cache=False)
        dspy.configure(lm=lm)
        if cot:
            self.find_incorrect_spans = dspy.ChainOfThought(FindIncorrectSpans)
        else:
            self.find_incorrect_spans = dspy.Predict(FindIncorrectSpans)

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, str | float]]:
        response = self.find_incorrect_spans(context=context, question=question, answer=answer)
        
        if logger:
            logger.info(f"====== Context ======:\n{context}\n")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Result ======:")
            for span, conf in zip(response.incorrect_spans, response.confidence):
                logger.info(f"- {span} ({conf})")
        
        if self.parse_span == "string_match" or self.parse_span is None:
            results = self._get_soft_label(answer, response.incorrect_spans, response.confidence)
        elif self.parse_span == "string_match_no_order":
            results = self._get_soft_label_2(answer, response.incorrect_spans, response.confidence)
        elif self.parse_span == "max_substring":
            results = self._get_soft_label_3(answer, response.incorrect_spans, response.confidence)
        else:
            raise ValueError(f"Invalid parse_span: {self.parse_span}")
        return results

    @staticmethod
    def _get_soft_label(answer: str, spans: list[str], confs: list[float]) -> list[dict[str, str | float]]:
        result = []
        curr_idx = 0
        for span, conf in zip(spans, confs):
            start = answer.find(span, curr_idx)
            if start < 0:
                continue
            end = start + len(span)
            result.append({"start": start, "end": end, "prob": conf})
            curr_idx = end
        return result
    
    @staticmethod
    def _get_soft_label_2(answer: str, spans: list[str], confs: list[float]) -> list[dict[str, str | float]]:
        result = []
        for span, conf in zip(spans, confs):
            start = answer.find(span)
            if start < 0:
                continue
            end = start + len(span)
            result.append({"start": start, "end": end, "prob": conf})
        return result
    
    @staticmethod
    def _get_soft_label_3(answer: str, spans: list[str], confs: list[float]) -> list[dict[str, str | float]]:
        """ maximum substring match """
        result = []
        for span, conf in zip(spans, confs):
            start, end = max_substring_match(answer, span)
            # if (end - start) >= len(span.text) // 2:  # at least half of the span is matched
            result.append({"start": start, "end": end, "prob": conf})
        return result


class DSPyCoTLabeler(BaseLabeler):
    def __init__(self, model: str, **kwargs):
        self.labeler = DSPySimpleLabeler(model, cot=True, **kwargs)

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, str | float]]:
        return self.labeler.label(question, answer, context, logger)
