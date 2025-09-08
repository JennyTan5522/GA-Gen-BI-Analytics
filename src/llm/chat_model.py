from abc import ABC
from langchain_anthropic import ChatAnthropic
from config.logger import get_logger

logger = get_logger("chat_model")

class ChatModel(ABC):
    """
    Abstract base class for LLM model integrations.
    """
    def __init__(self, api_key: str, model_name: str, base_url: str = "", **kwargs):
        """
        Initialize the base LLM class with API key, model name, and base URL.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = kwargs.get('temperature', 0)
        self.max_tokens = kwargs.get('max_tokens', 8000)

class ClaudeClient(ChatModel):
    """
    Client for interacting with Anthropic Claude models.
    """
    def __init__(self, api_key: str, model_name: str, base_url: str = "", **kwargs):
        """
        Initialize the ClaudeClient with API key, model name, and base URL.
        """
        super().__init__(api_key, model_name, base_url, **kwargs)
        try:
            self.client = ChatAnthropic(
                api_key=self.api_key,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ClaudeClient: {e}")

def get_chat_model(chat_model_name: str, **kwargs) -> ChatModel:
    """
    Get the appropriate chat model instance based on model_name.
    """
    if chat_model_name.lower() == "claude":
        return ClaudeClient(**kwargs).client
    else:
        raise ValueError(f"Unsupported model name: {chat_model_name}")
    