from .base_labeler import BaseLabeler
from .context_free_labeler import ContextFreeSimpleLabeler, ContextFreeXMLLabeler, ContextFreeMinEditLabeler, \
    PerplexityReasoningLabeler
from .context_dep_labeler import ContextDepSimpleLabeler, ContextSummarizingLabeler, \
    ContextAtomicFactLabeler, CreateContextLabeler, ContextDepMinEditLabeler, ContextDepMinEditLabeler2, \
    ContextReasonLabeler
from .qa_context_dep_labler import QAContextDepMinEditLabeler, QAContextDepJsonLabeler
from .kg_labeler import KGSimpleLabeler
from .context_reason_dep_labeler import ContextSumReasonLabeler, ContextReasonLabeler
from .tool_based_labeler import ToolDepSimpleLabeler
from .dspy_labeler import DSPySimpleLabeler, DSPyCoTLabeler


def get_labeler(labeler_name: str, model: str, prompt_id: str, **kwargs) -> BaseLabeler:
    if 'logging' in kwargs:
        del kwargs['logging']
        
    match labeler_name:
        case "context_free_simple":
            return ContextFreeSimpleLabeler(model, prompt_id, **kwargs)
        case "context_free_xml":
            return ContextFreeXMLLabeler(model, prompt_id, **kwargs)
        case "context_dep_simple":
            return ContextDepSimpleLabeler(model, prompt_id, **kwargs)
        case "context_summarizing":
            return ContextSummarizingLabeler(model, prompt_id, **kwargs)
        case "context_atomic_fact":
            return ContextAtomicFactLabeler(model, **kwargs)
        case "kg_simple_labeler":
            return KGSimpleLabeler(model, **kwargs)
        case "reason_context_sum":
            return ContextSumReasonLabeler(model, prompt_id, **kwargs)
        case "reason_context":
            return ContextReasonLabeler(model, prompt_id, **kwargs)
        case"tool_dep_simple":
            return ToolDepSimpleLabeler(model, prompt_id, **kwargs)
        case "create_context":
            return CreateContextLabeler(model, prompt_id, **kwargs)
        case "context_free_min_edit":
            return ContextFreeMinEditLabeler(model, **kwargs)
        case "context_dep_min_edit":
            return ContextDepMinEditLabeler(model, **kwargs)
        case "context_dep_min_edit_2":
            return ContextDepMinEditLabeler2(model, **kwargs)
        case "qa_context_dep_min_edit":
            return QAContextDepMinEditLabeler(model, **kwargs)
        case "qa_context_dep_json":
            return QAContextDepJsonLabeler(model, **kwargs)
        case "context_reason":
            return ContextReasonLabeler(model, prompt_id, **kwargs)
        case "dspy_simple":
            return DSPySimpleLabeler(model, **kwargs)
        case "dspy_cot":
            return DSPyCoTLabeler(model, **kwargs)
        case "perplexity_reasoning":
            return PerplexityReasoningLabeler(model, **kwargs)
        case _:
            raise ValueError(f"Labeler {labeler_name} not found")
