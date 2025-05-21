from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import asyncio

EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-embedding"
model = SentenceTransformer(EMBEDDING_MODEL_NAME,trust_remote_code=True)

def tokenize_text(text):
    return tokenize(text)

async def embed_text(text):
    # Run CPU-intensive embedding in a thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, model.encode, text)
