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
        Tạo (hoặc recreate) collection mới trên Qdrant và thêm chỉ mục payload cho file_name.
        """
        # Tạo collection với cấu hình vector
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dimension,
                distance=models.Distance.COSINE,
            )
        )
        print(f"Collection '{self.collection_name}' đã được tạo lại thành công!")

        # Thêm chỉ mục keyword cho file_name
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="file_name",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print(f"Chỉ mục keyword cho 'file_name' đã được tạo!")

        # Định nghĩa schema cho các trường khác (không cần chỉ mục)
        for field in ["content", "link_file"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.TEXT
            )
        print(f"Schema cho 'content' và 'link_file' đã được định nghĩa!")

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

    def upsert_points(self, encode_fields, payload_fields, batch_size=100):
        """
        Tạo và upsert các điểm dữ liệu theo batch để tránh timeout.
        """
        embeddings = self.encode_chunks(encode_fields)
        self.points = []
        
        for idx, chunk in enumerate(self.json_data):
            point = models.PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={field: chunk.get(field) for field in payload_fields}
            )
            self.points.append(point)
        
        # Chia nhỏ và upsert theo batch
        total_points = len(self.points)
        for i in range(0, total_points, batch_size):
            batch = self.points[i:i+batch_size]
            print(f"Upsert batch {i//batch_size + 1}/{(total_points+batch_size-1)//batch_size}: {len(batch)} điểm")
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True  # Đảm bảo hoàn thành trước khi tiếp tục
            )
        
        print(f"Đã upsert tổng cộng {total_points} điểm vào Qdrant thành công!")

    def search(self, query_text, limit=5):
        """
        Tìm kiếm trên collection với truy vấn được mã hóa.
        """
        query_embedding = self.model.encode(query_text)
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        return search_results

if __name__ == "__main__":
    # Đọc file JSON
    with open("ThuTucHanhChinh_DienBien.json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Khai báo thông tin Qdrant và collection
    collection_name = "dienbien"
    
    # Tạo đối tượng và sử dụng các phương thức
    collection_creator = VectorDBCollectionCreator(
        json_data=json_data,
        collection_name=collection_name,
    )

    # 1. Tạo collection trên Qdrant và thêm chỉ mục
    collection_creator.create_collection()

    # 2. Mã hóa các chunk và upsert vào Qdrant
    collection_creator.upsert_points(
        encode_fields=["file_name", "content"],
        payload_fields=["file_name", "content", "link_file"]
    )

    # 3. Thực hiện tìm kiếm thử
    query_text = "Thủ tục cấp giấy chứng nhận quyền sử dụng đất"
    results = collection_creator.search(query_text, limit=5)

    # Hiển thị kết quả tìm kiếm
    for i, result in enumerate(results):
        print(f"Kết quả {i+1}:")
        print(f"Điểm tương đồng: {result.score}")
        print(f"Tên tài liệu: {result.payload.get('file_name')}")
        print(f"Nội dung: {result.payload.get('content')}")
        print(f"Liên kết: {result.payload.get('link_file')}")
        print("-" * 50)