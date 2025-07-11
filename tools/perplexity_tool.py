from openai import OpenAI
import os

def call_perplexity(query:str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an asistant who helps the answer the user in a detailed manner."
            ),
        },
        {   
            "role": "user",
            "content": (
                query
            ),
        },
    ]

    client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")

    # chat completion without streaming
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )
    return response.choices[0].message.content
