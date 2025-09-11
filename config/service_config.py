from typing import Optional
from datetime import date
from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class ServiceConfig(BaseSettings):
    """
    Service configuration settings for the application.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_settings="utf-8",
        extra="ignore",
        case_sensitive=True
    )

    # Logging configuration
    LOG_LEVEL: str = Field("DEBUG", validation_alias="LOG_LEVEL")
    LOG_FILE: str = Field(f"logs/{date.today().isoformat()}_log.log", validation_alias="LOG_FILE")

    # Qdrant
    QDRANT_HOST: str = Field(..., validation_alias="QDRANT_HOST")
    QDRANT_API_KEY: SecretStr = Field(..., validation_alias="QDRANT_API_KEY")

    # Google Big Query
    GOOGLE_SERVICE_ACCOUNT_FILE: Optional[str] = Field(default=None, validation_alias="GOOGLE_SERVICE_ACCOUNT_FILE")

    # Table schema info path
    TABLE_SCHEMA_INFO_PATH: Optional[str] = Field(default="data/table_info", validation_alias="TABLE_SCHEMA_INFO_PATH")

    @field_validator("QDRANT_HOST")
    def validate_urls(cls, v: SecretStr):
        """Validate that the QDRANT_HOST is a URL starting with http or https."""
        value = v.get_secret_value() if isinstance(v, SecretStr) else v
        if not value.startswith(("http", "https")):
            raise ValueError("QDRANT_HOST must be a valid URL starting with http or https")
        return v

    def get_qdrant_api_key(self) -> str:
        """Return the Qdrant API key as a string."""
        try:
            return self.QDRANT_API_KEY.get_secret_value()
        except Exception as e:
            # Handle missing or invalid secret value
            raise RuntimeError(f"Failed to retrieve Qdrant API key: {e}")