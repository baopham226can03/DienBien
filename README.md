# HUIT Q&A API

HUIT Q&A API là một hệ thống trả lời câu hỏi tự động về thông tin tuyển sinh của Trường Đại học Công nghiệp TP.HCM (HUIT). API sử dụng **cache** để trả lời nhanh các câu hỏi phổ biến và **RAG (Retrieval-Augmented Generation)** để xử lý các câu hỏi phức tạp, kết hợp với lịch sử hội thoại để duy trì ngữ cảnh.

## Cấu trúc hệ thống
- **Cache**:
  - Lưu trữ 710+ câu hỏi phổ biến từ file Excel `BO_250_CAU_HOI_CUA_SINH_VIEN.xlsx`.
  - Sử dụng Qdrant local (`question_collection`) để tìm kiếm câu hỏi tương tự dựa trên embedding.
  - Model: `keepitreal/vietnamese-sbert` cho embedding tiếng Việt.
- **RAG**:
  - Tìm kiếm tài liệu từ collection `cam_nang_tuyen_sinh_HUIT_2025` (171 points) trên Qdrant cloud.
  - Kết hợp OpenAI (`gpt-4o-mini`) để tạo câu trả lời chi tiết.
- **History**:
  - Lưu lịch sử hội thoại theo `session_id` để duy trì ngữ cảnh.
- **API**:
  - Được xây dựng bằng FastAPI, chạy trên `http://0.0.0.0:8000`.
  - Endpoint chính: `POST /process`.

## Yêu cầu
- **Hệ điều hành**: Windows, Linux, hoặc macOS.
- **Python**: 3.8+.
- **Docker**: Để chạy Qdrant local.
- **API Keys**:
  - `OPENAI_API_KEY`: Cho OpenAI (`gpt-3.5-turbo`).
  - `QDRANT_API_KEY`: Cho Qdrant cloud.
- **Dung lượng**: ~2GB RAM cho Qdrant và `sentence-transformers`.

## Cài đặt
1. **Clone repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Tạo file `.env`**:
  Tạo file `.env` trong thư mục gốc với nội dung:
  ```
  # Qdrant configuration (RAG)
  export QDRANT_API_KEY=your_key
  export QDRANT_URL=your_url
  export QDRANT_PORT=6333
  QDRANT_COLLECTION=cam nang tuyen sinh HUIT 2025

  # Qdrant configuration (Cache)
  QDRANT_HOST=localhost
  QDRANT_CACHE_PORT=6333
  QDRANT_CACHE_COLLECTION=question_collection
  QDRANT_TOP_K=5

  # OpenAI configuration
  OPENAI_API_KEY=your_key
  OPENAI_MODEL_NAME=gpt-4o-mini

  # Model configuration
  LLM_MODEL_NAME=gpt-4o-mini
  EMBEDDING_MODEL_NAME=dangvantuan/vietnamese-document-embedding

  # Retriever configuration (RAG)
  RETRIEVER_K=10
  RETRIEVER_SCORE_THRESHOLD=0.5
  MAX_REWRITE_ROUNDS=3

  # Data paths
  ABBREVIATION_FILE=data/viettat.xlsx
  QUESTION_FILE=data/BỘ 250 CÂU HỎI CỦA SINH VIÊN VỀ TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP.HCM.xlsx
   ```

3. **Cài đặt Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   File `requirements.txt` nên có:
   ```
   fastapi
   uvicorn
   pandas
   openai
   qdrant-client
   sentence-transformers
   pyvi
   python-dotenv
   langchain
   langchain-openai
   httpx
   ```

4. **Chạy Qdrant local**:
   ```bash
   docker run -d -p 6333:6333 qdrant/qdrant
   ```
   Kiểm tra Qdrant:
   ```bash
   curl http://localhost:6333
   ```

## Chạy dự án
1. **Khởi động API**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   - `--reload`: Tự động restart khi sửa code (cho dev).
   - Log sẽ hiển thị:
     ```
     INFO: Application startup complete.
     INFO: Cache collection question_collection exists with 710 points.
     INFO: Collection cam_nang_tuyen_sinh_HUIT_2025 already has 171 points.
     ```

2. **Test API**:
   Dùng `curl` để gửi câu hỏi:
   ```bash
   curl -X POST "http://localhost:8000/process" -H "Content-Type: application/json" -d "{\"question\": \"Ngành nào điểm cao nhất?\"}"
   ```
   Response mẫu:
   ```
   Câu hỏi phù hợp trong cache: Ngành nào có điểm chuẩn cao nhất tại HUIT?
   Trả lời: Ngành Công nghệ thông tin có điểm chuẩn cao nhất, thường dao động từ 26-27 điểm.
   ```
   Hoặc (nếu cache miss):
   ```
   Không tìm thấy trong cache, xử lý bằng RAG
   Trả lời: Ngành Công nghệ thông tin tại HUIT có điểm chuẩn cao nhất...
   ```

## API Endpoint
- **POST `/process`**:
  - **Input**:
    ```json
    {
      "question": "Ngành nào điểm cao nhất?"
    }
    ```
    - Header (tùy chọn): `session-id` để duy trì lịch sử hội thoại.
  - **Output**:
    - Cache hit: Trả về câu hỏi khớp và câu trả lời.
    - Cache miss: Dùng RAG để trả lời.
    - Lỗi: JSON với `error` và mã trạng thái (400, 500).
  - **Media type**: `text/plain` (stream).

## Dữ liệu
- **Excel**: `data/xlsx/BO_250_CAU_HOI_CUA_SINH_VIEN.xlsx`
  - Cột: `STT`, `Câu hỏi`, `Câu trả lời`.
  - Dùng để tạo cache (`question_collection`).
- **Qdrant**:
  - Local: `question_collection` (710 points).
  - Cloud: `cam_nang_tuyen_sinh_HUIT_2025` (171 points).

## Ghi chú
- **Debug**: Xem log trong terminal hoặc thêm logging vào file:
  ```python
  logging.basicConfig(handlers=[logging.FileHandler("api.log"), logging.StreamHandler()])
  ```
- **Tối ưu cache**: Thêm câu hỏi phổ biến vào Excel để tăng tỷ lệ cache hit.
- **Mạng**: Đảm bảo kết nối ổn định tới OpenAI và Qdrant cloud.

## Liên hệ
- Có vấn đề? Liên hệ "sếp" qua <your-email> hoặc mở issue trên repo! 😄