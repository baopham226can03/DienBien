import os
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

from dotenv import load_dotenv
load_dotenv()


class VectorDBCollectionCreator:
    def __init__(self, json_data, collection_name,
                 model_name="dangvantuan/vietnamese-document-embedding", trust_remote_code=True):
        """
        Khởi tạo đối tượng với thông tin JSON, tên collection, thông tin kết nối Qdrant và model embedding.
        """
        self.json_data = json_data
        self.collection_name = collection_name
        self.qdrant_url = os.environ.get("QDRANT_URL")
        self.port = os.environ.get("QDRANT_PORT")
        self.api_key = os.environ.get("QDRANT_API_KEY")
        
        # Khởi tạo SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        # Kết nối đến Qdrant
        self.client = QdrantClient(url=self.qdrant_url, port=self.port, api_key=self.api_key)
        self.points = []

    def create_collection(self):
        """
        Tạo (hoặc recreate) collection mới trên Qdrant với cấu hình vector đã cho.
        """
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dimension,
                distance=models.Distance.COSINE,
            )
        )
        print(f"Collection '{self.collection_name}' đã được tạo lại thành công!")

    def encode_chunks(self, encode_fields):
        """
        Mã hóa từng đoạn trong danh sách JSON.
        """
        embeddings = []
        for chunk in self.json_data:
            text_to_encode = ""
            for field in encode_fields:
                text_to_encode += f"{chunk.get(field, '')}"
            embedding = self.model.encode(text_to_encode)
            embeddings.append(embedding)
        return embeddings

    def upsert_points(self, encode_fields, payload_fields):
        """
        Tạo các điểm dữ liệu từ các chunk và upsert vào Qdrant.
        """
        embeddings = self.encode_chunks(encode_fields)
        
        for idx, chunk in enumerate(self.json_data):
            point = models.PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={field: chunk.get(field) for field in payload_fields}
            )
            self.points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=self.points
        )
        print("Đã upsert các điểm vào Qdrant thành công!")


if __name__ == "__main__":
    # Đọc file JSON - sử dụng open() để mở file
    with open("Industry Admission Combinations.json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)  # Dùng load() cho file

    # Khai báo thông tin Qdrant và collection
    collection_name = "diem_chuan_xet_tuyen_HUIT_2025"

    # Tạo đối tượng và sử dụng các phương thức
    collection_creator = VectorDBCollectionCreator(
        json_data=json_data,
        collection_name=collection_name,
    )

    # 1. Tạo collection trên Qdrant
    collection_creator.create_collection()

    # 2. Mã hóa các chunk và upsert vào Qdrant
    collection_creator.upsert_points(encode_fields=["title", "text"], 
                                     payload_fields=["title", "text", "page", "word_count"])

    # 3. Thực hiện tìm kiếm
    query_text = "Điểm chuẩn ngành Công nghệ thực phẩm 2024?"
    results = collection_creator.search(query_text, limit=5)

    # Hiển thị kết quả tìm kiếm
    for i, result in enumerate(results):
        print(f"Kết quả {i+1}:")
        print(f"Điểm tương đồng: {result.score}")
        # print(f"Title: {result.payload.get('title')}")
        print(f"Nội dung: {result.payload.get('text')}")
        print(f"Metadata: {result.payload.get('metadata')}")
        # print(f"Trang: {result.payload.get('page')}")
        print("-" * 50)
