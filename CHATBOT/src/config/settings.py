from dotenv import load_dotenv
import os

load_dotenv()

# Qdrant configuration
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_RAG_COLLECTIONS = os.getenv("QDRANT_RAG_COLLECTIONS", "").split(",")
QDRANT_CACHE_COLLECTION = os.getenv("QDRANT_CACHE_COLLECTION", "du_lieu_lop_1")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", 5))

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Model configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "dangvantuan/vietnamese-document-embedding")

# Retriever configuration
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 10))
RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", 0.5))
MAX_REWRITE_ROUNDS = int(os.getenv("MAX_REWRITE_ROUNDS", 3))
LOP_1_SCORE_THRESHOLD=float(os.getenv("LOP_1_SCORE_THRESHOLD", 0.9))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0))
AGENT_TYPE = os.getenv('AGENT_TYPE', 'tool_calling')
DIEM_CHUAN_DATA_PATH = os.getenv('DIEM_CHUAN_DATA_PATH', 'src/utils/diem_chuan.csv')