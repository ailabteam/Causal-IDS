# Lưu file này với tên detect_anomalies_enhanced.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data'
RESULTS_DIR = 'results'
TRAIN_BENIGN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
GRAPH_FILE = os.path.join(RESULTS_DIR, 'causal_graph_ges_enhanced.txt')
SCALER_FILE = os.path.join(RESULTS_DIR, 'scaler_enhanced.joblib')
MODELS_DIR = os.path.join(RESULTS_DIR, 'causal_models_enhanced')
ANOMALY_SCORES_FILE = os.path.join(RESULTS_DIR, 'anomaly_scores_enhanced.csv')

def train_causal_mechanism():
    print("--- [Enhanced] Bắt đầu huấn luyện Cơ chế Nhân quả ---")
    df_train_orig = pd.read_csv(TRAIN_BENIGN_FILE)
    causal_graph = np.loadtxt(GRAPH_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    # Huấn luyện mô hình trên dữ liệu đã chuẩn hóa
    df_train_scaled = pd.DataFrame(scaler.transform(df_train_orig), columns=df_train_orig.columns)
    
    features = df_train_scaled.columns.tolist()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for i, target_feature in enumerate(features):
        safe_target_name = target_feature.replace('/', '_').replace(' ', '')
        model_filename = os.path.join(MODELS_DIR, f'model_{safe_target_name}.joblib')
        parents_indices = np.where(causal_graph[:, i] == 1)[0]
        
        if len(parents_indices) == 0:
            mean_value = df_train_scaled[target_feature].mean()
            joblib.dump(mean_value, model_filename)
        else:
            parent_features = [features[p_idx] for p_idx in parents_indices]
            X_train = df_train_scaled[parent_features]
            y_train = df_train_scaled[target_feature]
            # Sử dụng Gradient Boosting
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42, verbose=0)
            model.fit(X_train, y_train)
            joblib.dump(model, model_filename)
    print("--- Hoàn thành huấn luyện mô hình nâng cao. ---")
    return True

def calculate_anomaly_scores():
    print("--- [Enhanced] Bắt đầu tính toán Điểm Bất thường ---")
    features = pd.read_csv(TRAIN_BENIGN_FILE, nrows=0).columns.tolist()
    causal_graph = np.loadtxt(GRAPH_FILE)
    scaler = joblib.load(SCALER_FILE)
    
    all_scores_dfs = []
    chunksize = 100000
    
    reader = pd.read_csv(TEST_DATA_FILE, chunksize=chunksize, engine='python')
    for chunk_idx, df_test_chunk_orig in enumerate(reader):
        print(f"   - Đang xử lý khối {chunk_idx + 1}...")
        
        # Chuẩn hóa dữ liệu test chunk
        df_test_chunk_scaled = pd.DataFrame(scaler.transform(df_test_chunk_orig), columns=features)
        
        total_squared_errors = np.zeros(len(df_test_chunk_scaled))
        
        for i, target_feature in enumerate(features):
            safe_target_name = target_feature.replace('/', '_').replace(' ', '')
            model_path = os.path.join(MODELS_DIR, f'model_{safe_target_name}.joblib')
            model = joblib.load(model_path)
            parents_indices = np.where(causal_graph[:, i] == 1)[0]
            
            if len(parents_indices) == 0:
                predictions = np.full(len(df_test_chunk_scaled), model)
            else:
                parent_features = [features[p_idx] for p_idx in parents_indices]
                X_test_chunk = df_test_chunk_scaled[parent_features]
                predictions = model.predict(X_test_chunk)
                
            actual_values = df_test_chunk_scaled[target_feature]
            squared_errors = (actual_values - predictions) ** 2
            total_squared_errors += squared_errors

        chunk_anomaly_scores = np.sqrt(total_squared_errors / len(features))
        
        safe_scores = np.nan_to_num(chunk_anomaly_scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        df_to_append = pd.DataFrame({'score': safe_scores})
        all_scores_dfs.append(df_to_append)

    final_scores_df = pd.concat(all_scores_dfs, ignore_index=True)
    final_scores_df.to_csv(ANOMALY_SCORES_FILE, index=False)
    print(f"-> Đã lưu thành công {len(final_scores_df)} điểm bất thường vào file: {ANOMALY_SCORES_FILE}")

if __name__ == "__main__":
    if train_causal_mechanism():
        calculate_anomaly_scores()
