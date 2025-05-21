from qdrant_client import QdrantClient
from src.config.settings import QDRANT_URL, QDRANT_PORT, QDRANT_API_KEY

def init_qdrant_client():
    return QdrantClient(url=QDRANT_URL, port=QDRANT_PORT, api_key=QDRANT_API_KEY)