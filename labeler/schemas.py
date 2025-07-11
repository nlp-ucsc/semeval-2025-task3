from pydantic import BaseModel


class IncorrectSpan(BaseModel):
    text: str
    probability: float


class ResponseModel(BaseModel):
    incorrect_spans: list[IncorrectSpan]


class FactVerification(BaseModel):
    fact: str
    reasoning: str
    determination: bool
    confidence: float


class FactVerificationResponse(BaseModel):
    facts: list[FactVerification]


class ReasonModel(BaseModel):
    reasoning_steps: list[str]
    incorrect_spans: list[IncorrectSpan]
