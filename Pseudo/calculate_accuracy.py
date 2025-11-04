import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def calculate_accuracy():
    try:
        # Đọc file CSV
        df = pd.read_csv('du_pseudo_phi3.csv')
        dff = pd.read_csv('train_unlabeled_backup.csv')
        # Lấy toàn bộ dữ liệu
        labeled_data = df.copy()
        unlabeled_data = dff.copy()
        if labeled_data.empty:
            print("File dữ liệu trống!")
            return
            
        # Lấy nhãn thật và nhãn pseudo
        true_labels = unlabeled_data['label'].values
        pseudo_labels = labeled_data['pseudo_label'].values
        
        # Kiểm tra số lượng mẫu
        print(f"Số lượng mẫu có nhãn: {len(true_labels)}")
        
        # In phân bố nhãn thật
        print("\nPhân bố nhãn thật:")
        print(labeled_data['label'].value_counts().sort_index())
        
        # In phân bố nhãn pseudo
        print("\nPhân bố nhãn pseudo:")
        print(labeled_data['pseudo_label'].value_counts().sort_index())
        
        # Tính độ chính xác
        accuracy = accuracy_score(true_labels, pseudo_labels)
        print(f"\nĐộ chính xác: {accuracy:.4f}")
        
        # Lấy danh sách các nhãn duy nhất
        unique_labels = sorted(list(set(true_labels) | set(pseudo_labels)))
        
        # In báo cáo chi tiết
        print("\nBáo cáo chi tiết:")
        print(classification_report(true_labels, pseudo_labels, 
                                 labels=unique_labels,
                                 zero_division=0))
                                 
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")
        print("Vui lòng kiểm tra lại file dữ liệu và định dạng!")

if __name__ == "__main__":
    calculate_accuracy()