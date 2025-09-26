import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
import joblib

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data_subset'
RESULTS_DIR = 'results'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
ENCODING_DIM = 4 # Kích thước của lớp bottleneck (có thể tinh chỉnh)

# --- Định nghĩa mô hình Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
            # Không dùng Sigmoid/ReLU ở lớp cuối của decoder để nó có thể tái tạo giá trị âm/dương
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    print("--- Bắt đầu thực nghiệm với Baseline: Autoencoder ---")
    
    # 1. Tải và chuẩn bị dữ liệu
    print("-> Đang tải và chuẩn bị dữ liệu...")
    df_train_benign = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv'))
    df_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'))
    df_labels = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv'))
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train_benign)
    X_test = scaler.transform(df_test)

    # Chuyển dữ liệu sang PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_tensor = torch.FloatTensor(X_test).to(DEVICE)

    # 2. Khởi tạo và Huấn luyện mô hình
    print(f"-> Đang huấn luyện Autoencoder trên {DEVICE} trong {EPOCHS} epochs...")
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim=input_dim, encoding_dim=ENCODING_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in train_loader:
            inputs, = data
            inputs = inputs.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'   Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}')

    # 3. Tính toán điểm bất thường (lỗi tái tạo)
    print("-> Đang tính toán lỗi tái tạo trên tập test...")
    model.eval()
    with torch.no_grad():
        reconstructions = model(test_tensor)
        mse = nn.MSELoss(reduction='none')
        # Tính lỗi trên từng dòng (sample)
        anomaly_scores = mse(reconstructions, test_tensor).mean(axis=1).cpu().numpy()

    # 4. Đánh giá kết quả
    print("-> Đánh giá kết quả...")
    y_true = (df_labels['Label'] != 'BENIGN').astype(int)
    
    # Chọn ngưỡng tương tự các phương pháp khác (quantile 95 của lỗi trên tập train)
    # Để làm điều này, ta cần tính lỗi trên tập train trước
    train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    model.eval()
    with torch.no_grad():
        train_reconstructions = model(train_tensor)
        train_scores = mse(train_reconstructions, train_tensor).mean(axis=1).cpu().numpy()
    
    threshold = np.quantile(train_scores, 0.95)
    y_pred = (anomaly_scores > threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc_score = roc_auc_score(y_true, anomaly_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n--- KẾT QUẢ CHO AUTOENCODER ---")
    print("-----------------------------------")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    print(f"  AUC:       {auc_score:.4f}")
    print("-----------------------------------")

if __name__ == '__main__':
    main()
