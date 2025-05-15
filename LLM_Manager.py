import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
# from langchain_ollama import ChatOllama

class LLMManager:
    def __init__(self):
        self.llm = None
        load_dotenv()

    def initialize_claude_model(self):
        try:
            claude_api_key = os.getenv("ANTHROPIC_API_KEY")
            return ChatAnthropic(api_key=claude_api_key, model="claude-3-5-sonnet-20241022", temperature=0)
        except Exception as e:
            return None
    
    # def initialize_qwen_mode(self):
    #     return ChatOllama(model="qwen:7b", temperature=0)