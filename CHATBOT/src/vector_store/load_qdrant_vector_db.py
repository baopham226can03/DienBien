import os
import asyncio

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

class VectorDBCollectionRetriever:
    def __init__(self, collection_name, description, display_fields,
                 model_name="dangvantuan/vietnamese-document-embedding", trust_remote_code=True):
        """
        Khởi tạo đối tượng chỉ với tên collection, không cần JSON data
        """
        self.collection_name = collection_name
        self.description = description
        self.display_fields = display_fields
        self.qdrant_url = os.environ.get("QDRANT_URL")
        self.port = os.environ.get("QDRANT_PORT")
        self.api_key = os.environ.get("QDRANT_API_KEY")
        
        # Khởi tạo SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        
        # Kết nối đến Qdrant
        self.client = QdrantClient(url=self.qdrant_url, port=self.port, api_key=self.api_key)
    
    def init_from_qdrant(self):
        """
        Khởi tạo từ collection có sẵn trên Qdrant.
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            print(f"Đã kết nối với collection '{self.collection_name}' thành công!")
            return True
        except Exception as e:
            print(f"Không tìm thấy collection '{self.collection_name}': {str(e)}")
            return False
    
    async def search_with_threshold(self, query_text, limit=5, score_threshold=0.7):
        """
        Tìm kiếm trên collection với ngưỡng điểm tương đồng.
        """
        # Sử dụng to_thread để không chặn event loop
        query_embedding = await asyncio.to_thread(self.model.encode, query_text)

        search_results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold
        )
        
        documents = []
        for result in search_results:
            documents.append({
                "score": result.score,
                "payload": result.payload,
            })
        
        return documents
