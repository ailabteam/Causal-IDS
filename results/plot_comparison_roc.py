import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
from sklearn.preprocessing import StandardScaler # <<< THÊM DÒNG NÀY

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data_subset'
RESULTS_DIR = 'results'
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'final_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Đường dẫn đến các file cần thiết
LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv')

# Dictionary chứa thông tin về các mô hình để vẽ
MODELS_TO_PLOT = {
    'Causal-IDS (GBoost)': {
        'scores_file': os.path.join(RESULTS_DIR, 'anomaly_scores_enhanced.csv'),
        'color': 'red',
        'linestyle': '-'
    },
    'Autoencoder': {
        'scores_file': 'autoencoder_scores.csv', # Sẽ tạo file này
        'color': 'blue',
        'linestyle': '--'
    },
    'Isolation Forest': {
        'scores_file': 'isolation_forest_scores.csv', # Sẽ tạo file này
        'color': 'green',
        'linestyle': '-.'
    },
    'Causal-IDS (Linear)': {
        'scores_file': os.path.join(RESULTS_DIR, 'anomaly_scores.csv'),
        'color': 'purple',
        'linestyle': ':'
    }
}

def get_scores_from_baselines():
    """
    Hàm này chạy lại các baseline để lấy điểm bất thường của chúng.
    Chúng ta cần điểm, không chỉ là các chỉ số cuối cùng.
    """
    print("-> Đang lấy điểm từ các mô hình baseline...")
    
    # Tải dữ liệu chung
    df_train_benign = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv'))
    df_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'))
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train_benign)
    X_test = scaler.transform(df_test)

    # 1. Isolation Forest
    from sklearn.ensemble import IsolationForest
    print("   - Chạy Isolation Forest...")
    if_model = IsolationForest(contamination='auto', random_state=42)
    if_model.fit(X_train)
    # decision_function trả về điểm, điểm càng thấp càng bất thường.
    # Ta cần đảo dấu để điểm càng cao càng bất thường, cho nhất quán với các mô hình khác.
    if_scores = -if_model.decision_function(X_test)
    pd.DataFrame(if_scores, columns=['score']).to_csv(MODELS_TO_PLOT['Isolation Forest']['scores_file'], index=False)
    print(f"     -> Đã lưu điểm của Isolation Forest vào '{MODELS_TO_PLOT['Isolation Forest']['scores_file']}'")
    
    # 2. Autoencoder
    print("   - Chạy Autoencoder...")
    # (Copy lại code Autoencoder từ script baseline)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=4):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, encoding_dim), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(encoding_dim, 8), nn.ReLU(), nn.Linear(8, input_dim))
        def forward(self, x):
            return self.decoder(self.encoder(x))

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    model = Autoencoder(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(10): # Huấn luyện lại ngắn gọn
        for data in train_loader:
            inputs, = data; inputs = inputs.to(DEVICE)
            outputs = model(inputs); loss = criterion(outputs, inputs)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        reconstructions = model(test_tensor)
        ae_scores = nn.MSELoss(reduction='none')(reconstructions, test_tensor).mean(axis=1).cpu().numpy()
    pd.DataFrame(ae_scores, columns=['score']).to_csv(MODELS_TO_PLOT['Autoencoder']['scores_file'], index=False)
    print(f"     -> Đã lưu điểm của Autoencoder vào '{MODELS_TO_PLOT['Autoencoder']['scores_file']}'")

def main():
    """Hàm chính để vẽ biểu đồ ROC so sánh."""
    # Chạy lại baseline để có file điểm
    # Cần import StandardScaler ở đây vì nó được dùng trong hàm này
    from sklearn.preprocessing import StandardScaler
    get_scores_from_baselines()

    print("\n--- Bắt đầu vẽ biểu đồ ROC so sánh ---")
    
    # Tải nhãn thật
    df_labels = pd.read_csv(LABELS_FILE)
    y_true = (df_labels['Label'] != 'BENIGN').astype(int)

    plt.figure(figsize=(8, 8))

    # Vẽ đường ROC cho từng mô hình
    for model_name, model_info in MODELS_TO_PLOT.items():
        print(f"-> Đang xử lý: {model_name}")
        try:
            df_score = pd.read_csv(model_info['scores_file'])
            scores = df_score['score']
            
            # Làm sạch điểm (phòng trường hợp có NaN/inf)
            scores = pd.to_numeric(scores, errors='coerce').fillna(0)

            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=model_info['color'], linestyle=model_info['linestyle'], lw=2,
                     label=f'{model_name} (AUC = {roc_auc:.2f})')
        except FileNotFoundError:
            print(f"   [Warning] Không tìm thấy file điểm: {model_info['scores_file']}. Bỏ qua.")

    # Định dạng biểu đồ
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison for Intrusion Detection', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    # Lưu biểu đồ
    output_path = os.path.join(OUTPUT_DIR, 'roc_curve_comparison.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    
    print(f"\n--- Hoàn thành! ---")
    print(f"Biểu đồ so sánh đã được lưu tại: {output_path}")

if __name__ == '__main__':
    main()
