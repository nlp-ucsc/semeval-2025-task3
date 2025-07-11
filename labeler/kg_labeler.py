from typing import Annotated
from copy import deepcopy
import json
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
import networkx as nx
from pydantic import BaseModel, Field

from .base_labeler import BaseLabeler


class FalseFact(BaseModel):
    fact: str
    label: bool
    confidence: float


class AllFalseFacts(BaseModel):
    all_false_facts: list[FalseFact]


class TextSpan(BaseModel):
    text_span: str
    probability: float


class AllTextSpans(BaseModel):
    all_text_spans: list[TextSpan]


@tool
def get_information(
    entity: Annotated[str, "The entity to get information about"],
    graph: Annotated[nx.DiGraph, InjectedToolArg]
) -> str:
    """Gets information about a given entity from the graph.

    Returns:
        str: Information about the entity in the format:
            (<entity>, <relation_1>, <other_entity_1>)
            (<other_entity_2>, <relation_2>, <entity>)
            ...
    """
    if not graph.has_node(entity):
        return f"Entity '{entity}' not found in the knowledge graph."
    list_of_information = []
    for successor in graph.successors(entity):
        list_of_information.append(f"({entity}, {graph.edges[entity, successor]['relation']}, {successor})")
    for predecessor in graph.predecessors(entity):
        list_of_information.append(f"({predecessor}, {graph.edges[predecessor, entity]['relation']}, {entity})")
    return "\n".join(list_of_information)


def convert_to_json(all_false_facts: AllFalseFacts) -> str:
    all_facts = {'all_facts': []}
    for fact in all_false_facts.all_false_facts:
        if fact.label == True:
            continue
        all_facts['all_facts'].append({
            'fact': fact.fact,
            'probability': fact.confidence
        })
    return json.dumps(all_facts, indent=4)


def get_false_facts_to_spans_chain() -> Runnable:
    parse_false_facts_to_json_sys_prompt = dedent("""
        Map facts back to the original spans of text.

        You will be given a list of facts where each fact is associated with a label and a confidence score. Your task is to parse the facts that are labeled as "false" into JSON format with fields: "fact", "label", and "confidence".

        # Output Format
        The output must be in JSON format as shown below:

        ```json
        {{
        "all_false_facts": [
            {{
            "fact": "[fact_1]",
            "label": false,
            "confidence": [confidence_score_1]
            }},
            {{
            "fact": "[fact_2]",
            "label": false,
            "confidence": [confidence_score_2]
            }}
        ]
        }}
        ```

        # Notes
        - Only include the facts that are labeled as false. If there are no false facts, return the JSON with the "all_facts" field empty.
    """).strip()

    map_facts_to_spans_sys_prompt = dedent("""
        Find the spans of text that correspond to the given facts.

        You will be given a document and a json object. The json object contains fields: "fact" and "probability", where the facts are extracted from the given document. 

        Your task is to find spans of text from the document that correspond to the facts in the json object. Each fact can correspond to one or more text spans. You should make each span as short as possible and the text spans do not have to be complete sentences.

        # Output Format
        You must return your output in json format with 2 fields: "text_span" (span of text that corresponds to a fact) and "probability" (just copy the "probability" score in the input json object). Below is an example:

        ```json
        {{
        "all_text_spans": [
            {{
            "text_span": "[identified span]",
            "probability": [probability_score]
            }},
            {{
            "text_span": "[another identified span]",
            "probability": [probability_score]
            }}
        ]
        }}
        ```
    """).strip()

    map_facts_to_spans_user_prompt = dedent("""
        Document:
        ```
        {document}
        ```

        Facts in json format:
        ```json
        {all_facts}
        ```
    """).strip()
    parse_false_facts_to_json_prompt = ChatPromptTemplate.from_messages([
        ("system", parse_false_facts_to_json_sys_prompt),
        ("human", "{verification}"),
    ])
    map_facts_to_spans_prompt = ChatPromptTemplate.from_messages([
        ("system", map_facts_to_spans_sys_prompt),
        ("human", map_facts_to_spans_user_prompt),
    ])

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    llm_false_facts = llm.with_structured_output(AllFalseFacts, method='json_schema')
    llm_all_text_spans = llm.with_structured_output(AllTextSpans, method='json_schema')
    parse_false_facts_to_json_chain = parse_false_facts_to_json_prompt \
                                | llm_false_facts \
                                | convert_to_json
    map_facts_to_spans_chain = {
        "all_facts": parse_false_facts_to_json_chain,
        "document": lambda x: x['document']
    } | map_facts_to_spans_prompt | llm_all_text_spans

    return map_facts_to_spans_chain

class KGSimpleLabeler(BaseLabeler):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        fact_extraction_prompt = ChatPromptTemplate.from_messages(
            [("human", "Please break down the following sentence into independent facts: '''{sentence}'''")]
        )

        fact_verification_system_prompt = dedent("""
            You're a fact checker that uses knowledge graph tools to verify provided facts to determine their accuracy.

            You will be given a piece of text and a list of facts extracted from the text. To perform this task, you will need to use the provided `get_information` tool to get entities connected to the queried entity and their relations. A list of entities available to query will be provided for you.

            Once you have the information returned from the tool, use it to determine the truthfulness of the facts. For each fact, you should provide a label as either "true" or "false" and express how confident your are about your decision in a confidence score ranging from 0 and 1.

            # Notes
            - If a fact cannot be verified using the information returned information, mark it as "false".
        """).strip()

        fact_verification_user_prompt = dedent("""
            The text is:
            {answer}

            The facts are:
            {facts}

            The list of entities available to query are:
            {entities}

            Please verify all the facts and provide a label and confidence score for each fact.
        """).strip()


        fact_verification_prompt = ChatPromptTemplate.from_messages([
            ("system", fact_verification_system_prompt),
            ("human", fact_verification_user_prompt),
            ("placeholder", "{chat_history}"),
        ])
        self.fact_extraction_chain = fact_extraction_prompt | self.llm | StrOutputParser()
        llm_with_tools = self.llm.bind_tools([get_information])
        self.fact_verification_chain = fact_verification_prompt | llm_with_tools
        self.graph_transformer = LLMGraphTransformer(llm = self.llm)
        self.false_fact_to_span_chain = get_false_facts_to_spans_chain()

    def parse_to_graph(self, text: str) -> nx.DiGraph:
        doc = Document(text)
        graph_doc = self.graph_transformer.process_response(doc)
        
        graph = nx.DiGraph()
        graph.add_nodes_from([(node.id, {"type": node.type}) for node in graph_doc.nodes])
        graph.add_edges_from([(rel.source.id, rel.target.id, {"relation": rel.type}) for rel in graph_doc.relationships])
        return graph

    def label(self, question: str, answer: str, context: str, logger = None) -> list[dict[str, str | float]]:
        facts = self.fact_extraction_chain.invoke(answer)

        graph = self.parse_to_graph(context)
        entities = list(graph.nodes)

        history = []
        message = self.fact_verification_chain.invoke({"answer": answer, "facts": facts, "entities": entities})
        history.append(message)

        # print("\n=== Tool Calls and Responses ===")
        tool_call_logs = []
        for tool_call in message.tool_calls:
            tool_call_copy = deepcopy(tool_call)
            tool_call_copy['args']['graph'] = graph
            # print(f"\nQuerying entity: {tool_call_copy['args']['entity']}")
            tool_call_logs.append(f"** Querying entity: {tool_call_copy['args']['entity']}")
            tool_message = get_information.invoke(tool_call_copy)
            # print(f"Response:\n{tool_message.content}")
            tool_call_logs.append(f"-- Response:\n{tool_message.content}")
            history.append(tool_message)

        message = self.fact_verification_chain.invoke({"answer": answer, "facts": facts, "entities": entities, "chat_history": history})
        
        all_text_spans = self.false_fact_to_span_chain.invoke({"document": answer, "verification": message.content})
        soft_labels = self._get_soft_label(answer, all_text_spans.all_text_spans)
        
        if logger:
            logger.info("*****************************************")
            logger.info(f"====== Question ======:\n{question}\n")
            logger.info(f"====== Answer ======:\n{answer}\n")
            logger.info(f"====== Context ======:\n{context}\n")
            logger.info(f"====== Constructed Knowledge Graph ======:")
            logger.info(f"## Entities ##:")
            for node in graph.nodes(data=True):
                logger.info(f"- {(node[0], node[1].get('type'))}")
            logger.info("\n")
            logger.info(f"## Relations ##:")
            for edge in graph.edges(data=True):
                logger.info(f"- {(edge[0], edge[2]['relation'], edge[1])}")
            logger.info("\n")
            logger.info(f"====== Extracted Facts ======:\n{facts}\n")
            logger.info(f"====== Availabel Entities in KG ======:\n{entities}\n")
            logger.info(f"====== Tool Calls and Responses ======:")
            for log in tool_call_logs:
                logger.info(f"{log}")
            logger.info("\n")
            logger.info(f"====== Verification ======:\n{message.content}\n\n")
            logger.info(f"====== Mapping False Facts to Answer Spans ======:\n{all_text_spans.model_dump_json(indent=4)}\n")
            logger.info(f"====== Soft Labels ======:")
            for label in soft_labels:
                logger.info(f"- {answer[label['start']:label['end']]}")
            logger.info("\n")
        
        return soft_labels
    
    @staticmethod
    def _get_soft_label(answer: str, spans: list[TextSpan]) -> list[dict[str, str | float]]:
        result = []
        for span in spans:
            start = answer.find(span.text_span)
            if start < 0:
                continue
            end = start + len(span.text_span)
            result.append({"start": start, "end": end, "prob": span.probability})
        return result
        
        
