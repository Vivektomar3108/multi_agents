from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.mongo import init_db,close_db
from app.routes.pdf_routes import file_router
from app.routes.chat_router import chat_router
from app.routes.search_router import search_router
from app.routes.research_routes import router as research_routes
from app.routes.writer_agent_routes import router as writter_routes
from app.routes.social_media_router import social_media_router
import logging
import uvicorn
import dotenv
import os
# ─────────────────────────────────────────────
# 🌿 Environment Setup
# ─────────────────────────────────────────────
dotenv.load_dotenv()


logger = logging.getLogger("Resworb Research agent")


# ─────────────────────────────────────────────
# 🌍 FastAPI Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Initialize MongoDB
        # logger.info("🚀 Initializing MongoDB...")
        await init_db()
        # logger.info(f"✅ MongoDB initialized: {DB_NAME}")

        yield  # App runs while DB + Memory jobs active

    except Exception as e:
        logger.exception(f"❌ MongoDB initialization or memory setup failed: {e}")
        raise

    finally:
        # Close DB connection on shutdown
        try:
            logger.info("🛑 Closing MongoDB...")
            await close_db()
            logger.info("🔒 MongoDB connection closed.")
        except Exception as e:
            logger.exception(f"❌ MongoDB close failed: {e}")


app = FastAPI(title="Resworb Research Agent",
    description="Multi-Agent Resworb with FastAPI + MongoDB (Beanie ODM)",
    version="1.0.0",
    lifespan=lifespan,)


# ─────────────────────────────────────────────
# 🌐 CORS Middleware
# ─────────────────────────────────────────────
ENV = os.getenv("ENV", "development").lower()

if ENV == "development":
    origins = [
        "https://test-resworb.vercel.app",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://www.resworb.in",
        "https://resworb.in",
        "https://github.com",
        "http://127.0.0.1:5500",
    ]
else:
    origins = [
        "https://test-resworb.vercel.app",
        "https://www.resworb.in",
        "https://resworb.in",
        "http://127.0.0.1:5500",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(search_router)
app.include_router(file_router)
app.include_router(research_routes)
app.include_router(writter_routes)
app.include_router(social_media_router)


if __name__ == "__main__":
    
    ENV = os.getenv("ENV", "development").lower()
    is_dev = ENV in {"dev", "development", "local"}

    
    try:
        uvicorn.run(
            "app.main:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8001)),
            reload=is_dev,
            reload_dirs=["app"],
            workers=int(os.getenv("WORKERS", 1)),
            log_level="info",
        )
    except Exception as e:
        logger.exception(f"❌ Failed to start Resworb Research Agent API: {e}")