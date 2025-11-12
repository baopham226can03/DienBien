# SheepDog++

SheepDog++ là phiên bản nâng cấp của SheepDog, giữ nguyên khung cơ bản gồm **classifier**, **feature guidance**, và **contrastive learning**, nhưng cải tiến để nâng cao **chất lượng embedding**, **khả năng phân biệt lớp**, và **tính ổn định khi huấn luyện**.

## Các điểm cải tiến chính

1. **Contrastive Learning nâng cao với hard negative mining**  
   - Mô hình chủ động tìm các ví dụ “khó nhằn” (negative samples gần nhau về embedding nhưng khác nhãn) để tăng khả năng phân tách trong không gian embedding.

2. **Điều chỉnh trọng số loss và bổ sung consistency loss**  
   - Cân bằng các thành phần loss: cross-entropy, feature guidance, contrastive và KL divergence (consistency) để học đa khía cạnh mà không bị lệch trọng số.

3. **Projection head và embedding normalization**  
   - Sử dụng projection head sâu hơn với chuẩn hóa L2, giúp embeddings ổn định và dễ phân tách giữa các lớp.

Kết quả là SheepDog++ đạt **độ chính xác và F1 cao hơn** SheepDog gốc, đồng thời embeddings mạnh hơn và tổng quát hơn.
