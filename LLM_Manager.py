import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

class LLMManager:
    def __init__(self):
        self.llm = None
        load_dotenv()

    def initialize_claude_model(self):
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(api_key=claude_api_key, model="claude-3-5-sonnet-20241022", temperature=0)