from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ────────────────
    # 🌍 App Config
    # ────────────────
    app_env: str = Field("production", env="APP_ENV")
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")

    # ────────────────
    # 🔑 JWT / Auth
    # ────────────────
    jwt_secret: str = Field(..., env="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(60, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    jwt_leeway_seconds: int = Field(10, env="JWT_LEEWAY_SECONDS")

    # ────────────────
    # ⚙️ Worker Config
    # ────────────────
    worker_consumer_group: str = Field("agent_workers", env="WORKER_CONSUMER_GROUP")
    worker_consumer_name: str = Field("worker-1", env="WORKER_CONSUMER_NAME")

    # ────────────────
    # 🗄 MongoDB
    # ────────────────
    mongo_uri: str = Field("mongodb://localhost:27017", env="MONGO_URI")
    mongo_db_name: str = Field("resworb", env="MONGO_DB_NAME")
    mongo_user: str | None = Field(None, env="MONGO_USER")
    mongo_password: str | None = Field(None, env="MONGO_PASSWORD")
    mongo_auth_source: str = Field("admin", env="MONGO_AUTH_SOURCE")
    mongo_tls: bool = Field(False, env="MONGO_TLS")

    # ────────────────
    # 🚀 Redis
    # ────────────────
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_index: str = Field("vector_index", env="REDIS_INDEX")
    cache_type: str = Field("redis", env="CACHE_TYPE")

    # ────────────────
    # 📦 Vector DBs
    # ────────────────

    # Weaviate
    weaviate_url: str = Field("http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: str = Field("", env="WEAVIATE_API_KEY")
    weaviate_local: bool = Field(True, env="WEAVIATE_LOCAL")

    # ────────────────
    # 🤖 Groq AI
    # ────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_api_key_1: str = Field(..., env="GROQ_API_KEY_1")
    groq_api_key_2: str = Field(..., env="GROQ_API_KEY_2")
    groq_api_key_3: str = Field(..., env="GROQ_API_KEY_3")
    groq_api_key_4: str = Field(..., env="GROQ_API_KEY_4")
    groq_temperature: float = Field(0.1, env="GROQ_TEMPERATURE")
    groq_model_name: str = Field("llama-3.1-8b-instant", env="GROQ_MODEL_NAME")
    groq_streaming: bool = Field(True, env="GROQ_STREAMING")

    # ────────────────
    # 🔐 OAuth Providers
    # ────────────────
    google_client_id: str = Field("", env="GOOGLE_CLIENT_ID")
    google_client_secret: str = Field("", env="GOOGLE_CLIENT_SECRET")
    google_redirect_uri: str = Field("", env="GOOGLE_REDIRECT_URI")
    
    github_client_id: str = Field("", env="GITHUB_CLIENT_ID")
    github_client_secret: str = Field("", env="GITHUB_CLIENT_SECRET")
    github_redirect_uri: str = Field("", env="GITHUB_REDIRECT_URI")
    github_scope: str = Field("user:email read:user", env="GITHUB_SCOPE")

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
