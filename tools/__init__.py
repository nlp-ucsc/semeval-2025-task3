from .tools import search_gpt,search_you,search_perplexity,search_wikipedia

def get_tools() -> dict:
    return {
        "search_gpt": search_gpt,
        "search_you": search_you,
        "search_perplexity": search_perplexity,
        "search_wikipedia": search_wikipedia,
        "gpt":None,
    }
