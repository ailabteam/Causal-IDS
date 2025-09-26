import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data'
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'causal_models')

TRAIN_BENIGN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
GRAPH_FILE = os.path.join(RESULTS_DIR, 'causal_graph_ges.txt')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

ANOMALY_SCORES_FILE = os.path.join(RESULTS_DIR, 'anomaly_scores.csv')

def train_causal_mechanism():
    """
    Huấn luyện các mô hình hồi quy cho từng biến dựa trên các cha nhân quả của nó.
    """
    print("--- Bắt đầu Giai đoạn 4a: Huấn luyện Cơ chế Nhân quả ---")
    
    print("-> Đang tải dữ liệu huấn luyện và đồ thị nhân quả...")
    try:
        df_train = pd.read_csv(TRAIN_BENIGN_FILE)
        causal_graph = np.loadtxt(GRAPH_FILE)
    except FileNotFoundError as e:
        print(f"[Error] Không tìm thấy file cần thiết: {e}. Hãy chạy các bước trước.")
        return

    features = df_train.columns.tolist()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("-> Đang huấn luyện mô hình dự đoán cho từng biến...")
    for i, target_feature in enumerate(features):
        # <<< SỬA LỖI: Tạo tên file an toàn
        safe_target_name = target_feature.replace('/', '_').replace(' ', '')

        parents_indices = np.where(causal_graph[:, i] == 1)[0]
        
        model_filename = os.path.join(MODELS_DIR, f'model_{safe_target_name}.joblib')

        if len(parents_indices) == 0:
            mean_value = df_train[target_feature].mean()
            joblib.dump(mean_value, model_filename)
            print(f"   - {target_feature}: Không có cha, lưu giá trị trung bình.")
        else:
            parent_features = [features[p_idx] for p_idx in parents_indices]
            X_train = df_train[parent_features]
            y_train = df_train[target_feature]
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            joblib.dump(model, model_filename)
            print(f"   - {target_feature} <--- {parent_features}")
            
    print("\n--- Hoàn thành huấn luyện và đã lưu các mô hình. ---")

def calculate_anomaly_scores():
    """
    Tính điểm bất thường cho dữ liệu test.
    """
    print("\n--- Bắt đầu Giai đoạn 4b: Tính toán Điểm Bất thường ---")
    
    print("-> Đang tải dữ liệu test...")
    try:
        df_test = pd.read_csv(TEST_DATA_FILE)
    except FileNotFoundError as e:
        print(f"[Error] Không tìm thấy file test: {e}.")
        return

    features = df_test.columns.tolist()
    total_squared_errors = np.zeros(len(df_test))
    
    print("-> Đang tính toán lỗi tái tạo cho từng biến...")
    for i, target_feature in enumerate(features):
        # <<< SỬA LỖI: Tạo tên file an toàn
        safe_target_name = target_feature.replace('/', '_').replace(' ', '')
        
        model_path = os.path.join(MODELS_DIR, f'model_{safe_target_name}.joblib')
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"[Error] Không tìm thấy file mô hình cho {target_feature}. Hãy chạy phần huấn luyện trước.")
            continue

        causal_graph = np.loadtxt(GRAPH_FILE)
        parents_indices = np.where(causal_graph[:, i] == 1)[0]
        
        if len(parents_indices) == 0:
            mean_value = model
            predictions = np.full(len(df_test), mean_value)
        else:
            parent_features = [features[p_idx] for p_idx in parents_indices]
            X_test = df_test[parent_features]
            predictions = model.predict(X_test)
            
        actual_values = df_test[target_feature]
        squared_errors = (actual_values - predictions) ** 2
        total_squared_errors += squared_errors

    anomaly_scores = np.sqrt(total_squared_errors / len(features))
    
    print(f"-> Đang lưu {len(anomaly_scores)} điểm bất thường vào file...")
    pd.DataFrame(anomaly_scores, columns=['score']).to_csv(ANOMALY_SCORES_FILE, index=False)
    
    print(f"\n--- Hoàn thành tính toán. Điểm đã được lưu tại {ANOMALY_SCORES_FILE} ---")

if __name__ == "__main__":
    train_causal_mechanism()
    calculate_anomaly_scores()
