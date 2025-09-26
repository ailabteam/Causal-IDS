import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data_subset'
RESULTS_DIR = 'results'

TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
TEST_LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv')

def main():
    """
    Hàm chính để huấn luyện và đánh giá Isolation Forest.
    """
    print("--- Bắt đầu thực nghiệm với Baseline: Isolation Forest ---")
    
    # 1. Tải dữ liệu
    print("-> Đang tải dữ liệu...")
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_DATA_FILE)
        df_labels = pd.read_csv(TEST_LABELS_FILE)
    except FileNotFoundError as e:
        print(f"[LỖI] Không tìm thấy file dữ liệu: {e}.")
        return

    # 2. Chuẩn hóa dữ liệu
    # Isolation Forest hoạt động tốt hơn với dữ liệu đã chuẩn hóa
    print("-> Chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train)
    X_test = scaler.transform(df_test)
    
    # Tạo nhãn dạng số: 1 cho Attack, 0 cho Benign
    y_true = (df_labels['Label'] != 'BENIGN').astype(int)

    # 3. Huấn luyện mô hình Isolation Forest
    print("-> Đang huấn luyện Isolation Forest...")
    # `contamination` là tỷ lệ dự kiến của các điểm bất thường trong dữ liệu.
    # Chúng ta có thể ước tính nó từ tập test để có ngưỡng tốt.
    attack_ratio = y_true.mean()
    print(f"   Ước tính tỷ lệ tấn công (contamination): {attack_ratio:.4f}")
    
    # n_jobs=-1 để sử dụng tất cả các CPU core có sẵn
    model = IsolationForest(n_estimators=100, contamination=attack_ratio, random_state=42, n_jobs=-1)
    model.fit(X_train)
    
    # 4. Dự đoán và Đánh giá
    print("-> Đang dự đoán trên tập test và đánh giá...")
    
    # .predict() trả về 1 cho điểm bình thường (inlier), -1 cho bất thường (outlier)
    # Chúng ta cần chuyển nó về 0 (Benign) và 1 (Attack)
    predictions = model.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in predictions]

    # .decision_function() trả về điểm bất thường. Điểm càng thấp càng bất thường.
    # Chúng ta nhân với -1 để điểm càng cao càng bất thường (giống Causal-IDS)
    anomaly_scores = model.decision_function(X_test) * -1
    
    # Tính toán các chỉ số
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, anomaly_scores)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 5. In kết quả theo định dạng bảng
    print("\n--- KẾT QUẢ CHO ISOLATION FOREST ---")
    print("--------------------------------------")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print("--------------------------------------")

if __name__ == "__main__":
    main()
