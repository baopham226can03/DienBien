import logging

from src.vector_store.load_qdrant_vector_db import VectorDBCollectionRetriever
from src.utils.openai_utils import clarify_question, choose_best_match
from src.utils.embedder import embed_text
from src.utils.process_abbreviation import expand_abbreviations
from src.config.settings import QDRANT_CACHE_COLLECTION, QDRANT_TOP_K, LOP_1_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


class CacheAgent:
    def __init__(self):
        logger.info("Initializing CacheAgent with VectorDBCollectionRetriever...")
        self.retriever = VectorDBCollectionRetriever(
            collection_name='du_lieu_lop_1_2',
            description="Chứa thông tin về các cặp câu hỏi - trả lời của trường",
            display_fields=["question"],
            model_name="dangvantuan/vietnamese-embedding"
        )
        # Kiểm tra trạng thái collection
        try:
            collection_info = self.retriever.client.get_collection(QDRANT_CACHE_COLLECTION)
            logger.info(f"Collection {QDRANT_CACHE_COLLECTION} status: points_count={collection_info.points_count}, config={collection_info.model_dump()}")
            if collection_info.points_count == 0:
                logger.warning(f"Collection {QDRANT_CACHE_COLLECTION} is empty")
        except Exception as e:
            logger.error(f"Failed to access collection {QDRANT_CACHE_COLLECTION}: {str(e)}")

    async def run(self, question):
        """
        Kiểm tra câu hỏi trong cache và trả về câu trả lời nếu tìm thấy.

        Parameters:
        - question (str): Câu hỏi đầu vào.

        Returns:
        - tuple: (answer, matched_question) nếu tìm thấy trong cache, (None, None) nếu không.
        """
        logger.info(f"Processing cache for question: {question}")
        question = expand_abbreviations(question.lower().strip())
        clarified = await clarify_question(question.lower().strip())
        print(clarified)
        logger.info(f"Clarified question: {clarified}")

        # Tạo embedding cho câu hỏi
        try:
            query_vector = await embed_text(clarified)
            logger.info(f"Query vector created, length: {len(query_vector)}, sample: {query_vector[:5]}")
        except Exception as e:
            logger.error(f"Failed to create query vector: {str(e)}")
            return None, None

        # Tìm kiếm câu hỏi tương tự trong du_lieu_lop_1
        try:
            results = await self.retriever.search_with_threshold(clarified, limit=QDRANT_TOP_K, score_threshold=LOP_1_SCORE_THRESHOLD)
            logger.info(f"Search results count: {len(results)}")
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return None, None

        retrieved = []
        for result in results:
            payload = result.get("payload", {})
            score = result.get("score", 0.0)
            retrieved.append({
                "id": payload.get("id", 0),
                "question": payload.get("question", ""),
                "answer": payload.get("answer", "")
            })
            logger.info(f"Result: score={score}, payload={payload}")
        
        logger.info(f"Retrieved questions: {[r['question'] for r in retrieved]}")

        if len(retrieved) == 0:
            return None, None
        # Chọn câu hỏi phù hợp nhất
        try:
            best_id = await choose_best_match(clarified, retrieved)
            logger.info(f"Best match ID: {best_id}")
        except Exception as e:
            logger.error(f"Failed to choose best match: {str(e)}")
            return None, None

        if best_id == "0":
            logger.info("No matching question found in cache")
            return None, None

        # Tìm câu trả lời tương ứng
        for result in retrieved:
            if str(result["id"]) == best_id:
                logger.info(f"Cache hit: question={result['question']}, answer={result['answer']}")
                return result["answer"], result["question"]

        logger.info("No matching question found in cache after ID check")
        return None, None