# Lưu file này với tên learn_causal_graph_enhanced.py
import pandas as pd
import numpy as np
import os
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils
import time
from sklearn.preprocessing import StandardScaler
import joblib

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data_subset'
RESULTS_DIR = 'results'
TRAIN_BENIGN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
GRAPH_OUTPUT_FILE = os.path.join(RESULTS_DIR, 'causal_graph_ges_enhanced.txt')
GRAPH_IMAGE_FILE = os.path.join(RESULTS_DIR, 'causal_graph_ges_enhanced.png')
SCALER_FILE = os.path.join(RESULTS_DIR, 'scaler_enhanced.joblib')
SAMPLE_SIZE = 100000

def main():
    print("--- [Enhanced] Bắt đầu học Đồ thị Nhân quả ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_benign = pd.read_csv(TRAIN_BENIGN_FILE)
    
    print("-> Chuẩn hóa dữ liệu (Standardization)...")
    scaler = StandardScaler()
    df_benign_scaled = pd.DataFrame(scaler.fit_transform(df_benign), columns=df_benign.columns)
    
    print(f"-> Đang lưu đối tượng scaler vào file: {SCALER_FILE}")
    joblib.dump(scaler, SCALER_FILE)
    
    if len(df_benign_scaled) > SAMPLE_SIZE:
        df_sample = df_benign_scaled.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df_benign_scaled
        
    X = df_sample.to_numpy()
    feature_names = df_sample.columns.tolist()

    print("-> Bắt đầu chạy thuật toán GES trên dữ liệu đã chuẩn hóa...")
    start_time = time.time()
    Record = ges(X, score_func='local_score_BIC', maxP=None, parameters=None)
    W_est = Record['G'].graph
    end_time = time.time()
    print(f"   Hoàn thành GES trong {end_time - start_time:.2f} giây.")

    np.savetxt(GRAPH_OUTPUT_FILE, W_est, fmt='%d')
    print(f"-> Đã lưu đồ thị mới vào: {GRAPH_OUTPUT_FILE}")
    
    try:
        pydot_graph = GraphUtils.to_pydot(W_est, labels=feature_names)
        pydot_graph.write_png(GRAPH_IMAGE_FILE)
    except Exception as e:
        print(f"[Warning] Không thể tạo hình ảnh đồ thị. Lỗi: {e}")
    
    print("--- Hoàn thành học đồ thị phiên bản nâng cao! ---")

if __name__ == "__main__":
    main()
