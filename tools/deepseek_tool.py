import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def call_deepseek(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an assistant who helps the answer the user in a detailed manner.",
        },
        {"role": "user", "content": query},
    ]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-distill-llama-70b", messages=messages
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = "What is the capital of France?"
    print(call_deepseek(question))
