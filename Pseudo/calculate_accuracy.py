import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def calculate_accuracy():
    try:
        # Đọc file CSV chứa nhãn thật
        df_true = pd.read_csv('backup.csv')
        # Đọc file CSV chứa nhãn pseudo
        df_pseudo = pd.read_csv('unlabel.csv')
        # Kiểm tra số lượng mẫu
        if len(df_true) != len(df_pseudo):
            print("Lỗi: Số lượng mẫu trong hai file không khớp nhau!")
            return
            
        # Lấy nhãn thật và nhãn pseudo
        true_labels = df_true['label'].values
        pseudo_labels = df_pseudo['pseudo_label'].values
        
        # Tạo nhãn pseudo đảo ngược (0->1, 1->0)
        pseudo_labels_inverted = 1 - pseudo_labels
        
        # In thông tin cơ bản
        print(f"Tổng số mẫu: {len(true_labels)}")
        
        # In phân bố nhãn thật
        print("\nPhân bố nhãn thật (từ train_unlabeled_backup.csv):")
        print(df_true['label'].value_counts().sort_index())
        
        # In phân bố nhãn pseudo
        print("\nPhân bố nhãn pseudo gốc (từ du_pseudo_phi3.csv):")
        print(df_pseudo['pseudo_label'].value_counts().sort_index())
        
        # In phân bố nhãn pseudo đảo ngược
        print("\nPhân bố nhãn pseudo sau khi đảo (0->1, 1->0):")
        unique, counts = np.unique(pseudo_labels_inverted, return_counts=True)
        print(pd.Series(counts, index=unique).sort_index())
        
        # Tính độ chính xác cho cả 2 trường hợp
        accuracy_original = accuracy_score(true_labels, pseudo_labels)
        accuracy_inverted = accuracy_score(true_labels, pseudo_labels_inverted)
        
        print("\n=== KẾT QUẢ VỚI NHÃN PSEUDO GỐC ===")
        print(f"Độ chính xác: {accuracy_original:.4f}")
        print("\nBáo cáo chi tiết:")
        print(classification_report(true_labels, pseudo_labels, 
                                 labels=[0, 1],
                                 target_names=['Giả (0)', 'Thật (1)'],
                                 digits=4))
        
        print("\n=== KẾT QUẢ VỚI NHÃN PSEUDO ĐẢO NGƯỢC ===")
        print(f"Độ chính xác: {accuracy_inverted:.4f}")
        print("\nBáo cáo chi tiết:")
        print(classification_report(true_labels, pseudo_labels_inverted, 
                                 labels=[0, 1],
                                 target_names=['Giả (0)', 'Thật (1)'],
                                 digits=4))
        
        # Kết luận
        print("\n=== KẾT LUẬN ===")
        if accuracy_original > accuracy_inverted:
            print("Nhãn pseudo gốc cho kết quả tốt hơn!")
        elif accuracy_original < accuracy_inverted:
            print("Nhãn pseudo đảo ngược cho kết quả tốt hơn!")
        else:
            print("Cả hai cách cho kết quả như nhau!")
                                 
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")
        print("Vui lòng kiểm tra lại file dữ liệu và định dạng!")

if __name__ == "__main__":
    calculate_accuracy()