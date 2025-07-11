from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class Translator(object):
    def __init__(self, model, target_language: str):
        self.model = model
        self.target_language = target_language
        self.chat_model = ChatOpenAI(model=model)
        self.prompt = f"For ALL the following conversations, no matter what I send, the ONLY thing you need to reply is the {self.target_language} translation of what I send. DON'T do anything else, just give me the {self.target_language} translation"

    def translate(self, text):
        messages = [HumanMessage(content=self.prompt), SystemMessage(content=text)]
        response = self.chat_model.invoke(messages)
        return response.content
