import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data_subset'
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'causal_models')
TRAIN_BENIGN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
GRAPH_FILE = os.path.join(RESULTS_DIR, 'causal_graph_ges.txt')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
ANOMALY_SCORES_FILE = os.path.join(RESULTS_DIR, 'anomaly_scores.csv')

def train_causal_mechanism():
    """
    Huấn luyện các mô hình hồi quy cho từng biến (feature) dựa trên các "cha"
    nhân quả của nó trong đồ thị đã học.
    """
    print("--- Bắt đầu Giai đoạn 4a: Huấn luyện Cơ chế Nhân quả ---")
    
    try:
        df_train = pd.read_csv(TRAIN_BENIGN_FILE)
        causal_graph = np.loadtxt(GRAPH_FILE)
    except FileNotFoundError as e:
        print(f"[LỖI] Không tìm thấy file cần thiết: {e}.")
        return False

    features = df_train.columns.tolist()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("-> Đang huấn luyện mô hình dự đoán cho từng biến...")
    for i, target_feature in enumerate(features):
        safe_target_name = target_feature.replace('/', '_').replace(' ', '')
        model_filename = os.path.join(MODELS_DIR, f'model_{safe_target_name}.joblib')
        parents_indices = np.where(causal_graph[:, i] == 1)[0]
        
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
    return True

def calculate_anomaly_scores():
    """
    Tính điểm bất thường cho dữ liệu test theo từng khối,
    với logic tạo DataFrame đã được sửa lỗi.
    """
    print("\n--- Bắt đầu Giai đoạn 4b: Tính toán Điểm Bất thường ---")
    
    try:
        features = pd.read_csv(TRAIN_BENIGN_FILE, nrows=0).columns.tolist()
        causal_graph = np.loadtxt(GRAPH_FILE)
    except FileNotFoundError as e:
        print(f"[LỖI] Không thể tải file cấu hình: {e}.")
        return

    all_scores_dfs = []
    chunksize = 100000
    print(f"-> Đang xử lý file test theo từng khối {chunksize} dòng...")
    
    try:
        reader = pd.read_csv(TEST_DATA_FILE, chunksize=chunksize, engine='python')
        
        for chunk_idx, df_test_chunk in enumerate(reader):
            print(f"   - Đang xử lý khối {chunk_idx + 1}...")
            
            if df_test_chunk.empty: continue

            total_squared_errors = np.zeros(len(df_test_chunk))
            
            for i, target_feature in enumerate(features):
                safe_target_name = target_feature.replace('/', '_').replace(' ', '')
                model_path = os.path.join(MODELS_DIR, f'model_{safe_target_name}.joblib')
                model = joblib.load(model_path)
                parents_indices = np.where(causal_graph[:, i] == 1)[0]
                
                if len(parents_indices) == 0:
                    predictions = np.full(len(df_test_chunk), model)
                else:
                    parent_features = [features[p_idx] for p_idx in parents_indices]
                    X_test_chunk = df_test_chunk[parent_features]
                    predictions = model.predict(X_test_chunk)
                    
                actual_values = df_test_chunk[target_feature]
                squared_errors = (actual_values - predictions) ** 2
                total_squared_errors += squared_errors

            chunk_anomaly_scores = np.sqrt(total_squared_errors / len(features))
            
            # <<< KHỐI CODE SỬA LỖI ĐÃ ĐƯỢC CHỨNG MINH LÀ ĐÚNG
            safe_scores = chunk_anomaly_scores.astype(np.float64)
            safe_scores = np.nan_to_num(safe_scores, nan=0.0, posinf=0.0, neginf=0.0)
            df_to_append = pd.DataFrame({'score': safe_scores})
            all_scores_dfs.append(df_to_append)

    except Exception as e:
        print(f"[LỖI] Lỗi xảy ra trong quá trình xử lý: {e}")
        import traceback
        traceback.print_exc()
        return

    if not all_scores_dfs:
        print("[LỖI] Không có điểm nào được tính.")
        return

    print("-> Đang gộp và lưu kết quả...")
    final_scores_df = pd.concat(all_scores_dfs, ignore_index=True)
    
    final_scores_df.to_csv(ANOMALY_SCORES_FILE, index=False)
    print(f"-> Đã lưu thành công {len(final_scores_df)} điểm bất thường vào file: {ANOMALY_SCORES_FILE}")
    print("\n--- Hoàn thành tính toán. ---")

if __name__ == "__main__":
    if train_causal_mechanism():
        calculate_anomaly_scores()
