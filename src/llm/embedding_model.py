from abc import ABC, abstractmethod
# from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Base class for embedding models
class EmbeddingModel(ABC):
    @abstractmethod
    def embed_query(self, text: str):
        """Embed a single query string."""
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]):
        """Embed a list of document strings."""
        pass

# SBERT embedding implementation
class SBERTEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize SBERT embedding model with the given model name."""
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            # Handle model loading errors
            raise RuntimeError(f"Failed to load SBERT model '{model_name}': {e}")

    def embed_query(self, text: str):
        """Embed a single query string using SBERT."""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            raise RuntimeError(f"SBERT embedding failed for query: {e}")

    def embed_documents(self, texts: list[str]):
        """Embed a list of document strings using SBERT."""
        try:
            return self.model.encode(texts).tolist()
        except Exception as e:
            raise RuntimeError(f"SBERT embedding failed for documents: {e}")

# OpenAI embedding implementation
class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, **kwargs):
        """
        Initialize OpenAI embedding model with optional parameters.
        Supported kwargs: embedding_model, embedding_api_key, embedding_base_url.
        """
        try:
            model = kwargs.get('embedding_model', 'text-embedding-ada-002')
            api_key = kwargs.get('embedding_api_key', None)
            base_url = kwargs.get('embedding_base_url', None)
            self.embedding_model = OpenAIEmbeddings(model=model, api_key=api_key, base_url=base_url)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAIEmbeddings: {e}")

    def embed_query(self, text: str):
        """Embed a single query string using OpenAI embeddings."""
        try:
            # OpenAIEmbeddings expects a list of strings
            return self.embedding_model.embed_documents([text])[0]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed for query: {e}")

    def embed_documents(self, texts: list[str]):
        """Embed a list of document strings using OpenAI embeddings."""
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding failed for documents: {e}")

def get_embedding_model(embedding_model_name: str, **kwargs):
    """Factory function to get embedding model by name."""
    if embedding_model_name == 'sbert':
        # return SBERTEmbeddingModel(**kwargs)
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    elif embedding_model_name == 'openai':
        return OpenAIEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model_name}")