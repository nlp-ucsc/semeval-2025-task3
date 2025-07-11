from langchain_core.tools import tool
from .deploy_helper import query_system
from .perplexity_tool import call_perplexity
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


@tool
def search_gpt(query:str) -> str:
    """Search Engine to retrieve more information about a given query"""

    return query_system(query,"gpt-search")

@tool
def search_you(query:str) -> str:
    """Search Engine to retrieve more information about a given query"""

    return query_system(query,"you-research")

@tool
def search_perplexity(query:str) -> str:
    """Search Engine to retrieve more information about a given query"""

    return call_perplexity(query)

@tool
def search_wikipedia(query:str) -> str:
    """Search Engine to retrieve more information about a given query"""

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)
