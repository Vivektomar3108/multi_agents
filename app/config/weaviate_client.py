# app/config/weaviate_client.py

import weaviate
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, host="127.0.0.1", port=8080):
        if getattr(self, "_initialized", False):
            return

        self.host = host
        self.port = port
        self.grpc_port = 50051
        self.client = None
        self._initialized = True

        self._connect_with_retry()

    def _connect_with_retry(self, retries=10, delay=2):
        """Retry REST-only connection until ready."""

        for attempt in range(1, retries + 1):
            logger.info(f"[{attempt}/{retries}] Connecting to Weaviate REST {self.host}:{self.port}...")

            try:
                # 👇 REST-only mode (no grpc_port parameter passed)
                self.client = weaviate.connect_to_local(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    skip_init_checks=True
                )

                if self.client.is_live():
                    logger.info("🔥 Connected to Weaviate (REST mode, gRPC disabled).")
                    return

                logger.warning("⚠️ Weaviate REST reachable but not live yet... waiting...")

            except Exception as e:
                logger.warning(f"Connection error: {e}")

            time.sleep(delay)

        raise RuntimeError("🚨 Failed to connect to Weaviate REST API after retries.")

    def __getattr__(self, name):
        client = object.__getattribute__(self, "client")
        if client is None:
            raise RuntimeError("Weaviate client not initialized.")
        return getattr(client, name)
