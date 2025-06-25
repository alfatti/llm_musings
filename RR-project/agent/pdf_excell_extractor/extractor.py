### file: extractor.py
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import EXTRACTOR_PROMPT_PDF, EXTRACTOR_PROMPT_EXCEL


class Extractor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.prompts = {
            "pdf": EXTRACTOR_PROMPT_PDF,
            "excel": EXTRACTOR_PROMPT_EXCEL,
        }

    def extract(self, text: str, file_type: str) -> str:
        prompt = self.prompts[file_type].format(text=text)
        return self.llm.invoke(prompt).content
