import os
from typing import Optional, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.config.settings import QDRANT_CACHE_COLLECTION, QDRANT_TOP_K, LOP_1_SCORE_THRESHOLD


class CacheBot:
    def __init__(
        self,
        collection_name: str = QDRANT_CACHE_COLLECTION,
        embedding_model: str = 'dangvantuan/vietnamese-embedding',
        score_threshold: float = float(LOP_1_SCORE_THRESHOLD),
        qdrant_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initializes the CacheBot with Qdrant client and embedding model.

        Args:
            collection_name: Name of the Qdrant collection to search.
            embedding_model: Pretrained embedding model identifier.
            score_threshold: Minimum score required to consider a result valid.
            qdrant_url: URL for Qdrant; falls back to env var if not provided.
            api_key: API key for Qdrant; falls back to env var if not provided.
        """
        # Setup Qdrant client
        self.client = QdrantClient(
            url=qdrant_url or os.environ.get('QDRANT_URL'),
            api_key=api_key or os.environ.get('QDRANT_API_KEY')
        )

        # Load embedding model
        self.embedding = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.collection_name = collection_name
        self.score_threshold = score_threshold

    async def get_best_answer(
        self,
        query: str,
        limit: int = 10
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Searches the collection and returns the question and answer of the highest-scoring result
        if it exceeds the threshold. Otherwise, returns (None, None).

        Args:
            query: The query string to embed and search.
            limit: The maximum number of results to retrieve from Qdrant.

        Returns:
            A tuple of (question, answer) or (None, None).
        """
        print(f"Encoding query: {query}")
        query_vec = self.embedding.encode(query)

        print(f"Searching Qdrant with vector, limit={QDRANT_TOP_K}")
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            limit=QDRANT_TOP_K
        )

        if not results:
            print("No results returned from Qdrant.")
            return None, None

        best_point = max(results, key=lambda point: point.score)
        print(f"Best result score: {best_point.score}")
        print(f"Best question: {best_point.payload.get('question')}")

        if best_point.score > self.score_threshold:
            payload = best_point.payload or {}
            question = payload.get('question')
            answer = payload.get('answer')
            print(f"Returning answer with score above threshold ({self.score_threshold})")
            return question, answer

        print(f"No result exceeded threshold ({self.score_threshold})")
        return None, None
