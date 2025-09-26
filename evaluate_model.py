import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data_subset'
RESULTS_DIR = 'results'

LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv')
SCORES_FILE = os.path.join(RESULTS_DIR, 'anomaly_scores.csv')

OUTPUT_DIR = os.path.join(RESULTS_DIR, 'evaluation')

def main():
    """Hàm chính để đánh giá mô hình."""
    print("--- Bắt đầu Giai đoạn 5: Đánh giá Mô hình ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Tải nhãn thực tế và điểm bất thường
    print("-> Đang tải nhãn và điểm bất thường...")
    try:
        df_labels = pd.read_csv(LABELS_FILE)
        df_scores = pd.read_csv(SCORES_FILE)
    except FileNotFoundError as e:
        print(f"[Error] Không tìm thấy file cần thiết: {e}. Hãy chạy các bước trước.")
        return
    
    # <<< THÊM BƯỚC LÀM SẠCH ĐIỂM
    invalid_scores = df_scores['score'].isna().sum() + np.isinf(df_scores['score']).sum()
    if invalid_scores > 0:
        print(f"[Warning] Tìm thấy {invalid_scores} điểm bất thường không hợp lệ (NaN/inf).")
        median_score = df_scores['score'].median()
        df_scores['score'].fillna(median_score, inplace=True)
        df_scores.replace([np.inf, -np.inf], median_score, inplace=True)
        print(f"   Đã thay thế chúng bằng giá trị trung vị: {median_score:.2f}")
    # >>> KẾT THÚC BƯỚC LÀM SẠCH

    df_eval = pd.concat([df_labels, df_scores], axis=1)
    df_eval['is_attack'] = (df_eval['Label'] != 'BENIGN').astype(int)

    # 2. Phân tích phân phối điểm
    print("\n-> Phân tích phân phối điểm bất thường...")
    score_benign = df_eval[df_eval['is_attack'] == 0]['score']
    score_attack = df_eval[df_eval['is_attack'] == 1]['score']
    
    print(f"   Điểm trung bình (Benign): {score_benign.mean():.2f} +/- {score_benign.std():.2f}")
    print(f"   Điểm trung bình (Attack): {score_attack.mean():.2f} +/- {score_attack.std():.2f}")

    plt.figure(figsize=(10, 6))
    # Sử dụng clip để tránh các giá trị ngoại lai quá lớn làm hỏng biểu đồ
    upper_bound = score_attack.quantile(0.99) if not score_attack.empty else score_benign.quantile(0.99)
    sns.kdeplot(score_benign, label='Benign', fill=True, clip=(0, upper_bound))
    sns.kdeplot(score_attack, label='Attack', fill=True, clip=(0, upper_bound))
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Causal Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'score_distribution.png'), dpi=600, bbox_inches='tight')
    print(f"   Đã lưu biểu đồ phân phối điểm vào '{OUTPUT_DIR}/score_distribution.png' (DPI=600)")
    plt.close()

    # 3. Đánh giá với ngưỡng
    threshold = score_benign.quantile(0.95)
    print(f"\n-> Chọn ngưỡng (threshold) = {threshold:.2f} (quantile 95 của điểm Benign)")
    
    y_true = df_eval['is_attack']
    y_pred = (df_eval['score'] > threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"   False Positive Rate (FPR): {fpr:.4f}")
    
    # 4. Vẽ đường cong ROC
    print("\n-> Đang vẽ đường cong ROC...")
    fpr_roc, tpr_roc, _ = roc_curve(y_true, df_eval['score'])
    roc_auc = auc(fpr_roc, tpr_roc)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=600, bbox_inches='tight')
    print(f"   Đã lưu biểu đồ ROC vào '{OUTPUT_DIR}/roc_curve.png' (DPI=600)")
    plt.close()
    
    print("\n--- Hoàn thành Giai đoạn 5! ---")
    print("   Tất cả kết quả đánh giá đã được tạo.")

if __name__ == "__main__":
    main()
