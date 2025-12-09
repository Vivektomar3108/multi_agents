# app/config/setting.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # --------------------------
    # Pydantic v2 Global Settings
    # --------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",     # prevent accidental env injection
        case_sensitive=False
    )

    # ────────────────
    # 🌍 App Config
    # ────────────────
    app_env: str = Field(default="production", validation_alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=8000, validation_alias="APP_PORT")

    # ────────────────
    # 🔑 JWT / Auth
    # ────────────────
    jwt_secret: str = Field(validation_alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=60, validation_alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, validation_alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    jwt_leeway_seconds: int = Field(default=10, validation_alias="JWT_LEEWAY_SECONDS")

    # ────────────────
    # ⚙️ Worker Config
    # ────────────────
    worker_consumer_group: str = Field(default="agent_workers", validation_alias="WORKER_CONSUMER_GROUP")
    worker_consumer_name: str = Field(default="worker-1", validation_alias="WORKER_CONSUMER_NAME")

    # ────────────────
    # 🗄 MongoDB
    # ────────────────
    mongo_uri: str = Field(default="mongodb://localhost:27017", validation_alias="MONGO_URI")
    mongo_db_name: str = Field(default="resworb", validation_alias="MONGO_DB_NAME")
    mongo_user: str | None = Field(default=None, validation_alias="MONGO_USER")
    mongo_password: str | None = Field(default=None, validation_alias="MONGO_PASSWORD")
    mongo_auth_source: str = Field(default="admin", validation_alias="MONGO_AUTH_SOURCE")
    mongo_tls: bool = Field(default=False, validation_alias="MONGO_TLS")

    # ────────────────
    # 🚀 Redis
    # ────────────────
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    redis_index: str = Field(default="vector_index", validation_alias="REDIS_INDEX")
    cache_type: str = Field(default="redis", validation_alias="CACHE_TYPE")

    # ────────────────
    # 🤖 Groq AI
    # ────────────────
    groq_api_key: str = Field(validation_alias="GROQ_API_KEY")
    groq_api_key_1: str = Field(validation_alias="GROQ_API_KEY_1")
    groq_api_key_2: str = Field(validation_alias="GROQ_API_KEY_2")
    groq_api_key_3: str = Field(validation_alias="GROQ_API_KEY_3")
    groq_api_key_4: str = Field(validation_alias="GROQ_API_KEY_4")

    groq_temperature: float = Field(default=0.1, validation_alias="GROQ_TEMPERATURE")
    groq_model_name: str = Field(default="llama-3.1-8b-instant", validation_alias="GROQ_MODEL_NAME")
    groq_streaming: bool = Field(default=True, validation_alias="GROQ_STREAMING")

    # ────────────────
    # ☁️ AWS / S3 Config
    # ────────────────
    aws_access_key_id: str = Field(validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(validation_alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(default="ap-south-1", validation_alias="AWS_DEFAULT_REGION")
    s3_bucket: str = Field(validation_alias="S3_BUCKET")


settings = Settings()
