import pandas as pd
import numpy as np
import os
import joblib

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data'
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'causal_models')
TRAIN_BENIGN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
GRAPH_FILE = os.path.join(RESULTS_DIR, 'causal_graph_ges.txt')
TEST_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

# File output debug
DEBUG_CHUNK_OUTPUT_FILE = os.path.join(RESULTS_DIR, 'debug_chunk_0_scores.csv')

def debug_first_chunk():
    """
    Hàm debug chỉ xử lý khối dữ liệu đầu tiên và lưu kết quả,
    với logic tạo DataFrame đã được sửa lỗi.
    """
    print("--- Bắt đầu Debug ---")
    
    try:
        features = pd.read_csv(TRAIN_BENIGN_FILE, nrows=0).columns.tolist()
        causal_graph = np.loadtxt(GRAPH_FILE)
        print("-> Tải features và graph thành công.")
    except Exception as e:
        print(f"[LỖI] Không thể tải file cấu hình: {e}")
        return

    chunksize = 100000
    
    try:
        # Chỉ đọc khối đầu tiên
        reader = pd.read_csv(TEST_DATA_FILE, chunksize=chunksize, engine='python')
        df_test_chunk = next(reader)
        print(f"-> Đã đọc khối đầu tiên, kích thước: {df_test_chunk.shape}")

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
        
        # <<< BẮT ĐẦU KHỐI CODE SỬA LỖI, AN TOÀN HƠN
        # Bước 1: Ép kiểu numpy array về float64 để đảm bảo tính tương thích
        safe_scores = chunk_anomaly_scores.astype(np.float64)

        # Bước 2: Xử lý các giá trị NaN và Inf trên numpy array.
        # np.nan_to_num sẽ thay NaN bằng 0, inf và -inf bằng số rất lớn/nhỏ.
        # Để an toàn, chúng ta cũng có thể thay thế inf bằng 0.
        nan_count = np.isnan(safe_scores).sum()
        inf_count = np.isinf(safe_scores).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   [Debug] Tìm thấy {nan_count} NaN và {inf_count} Inf trong điểm số.")
            median_score = np.nanmedian(safe_scores[np.isfinite(safe_scores)]) # Tính trung vị của các số hợp lệ
            if np.isnan(median_score): median_score = 0 # Nếu tất cả đều không hợp lệ, dùng 0
            print(f"   [Debug] Sẽ thay thế chúng bằng giá trị trung vị: {median_score}")
            safe_scores = np.nan_to_num(safe_scores, nan=median_score, posinf=median_score, neginf=median_score)

        # Bước 3: Xây dựng DataFrame từ dictionary (cách an toàn nhất)
        df_to_write = pd.DataFrame({'score': safe_scores})
        # >>> KẾT THÚC KHỐI CODE SỬA LỖI

        # THÊM DEBUG CUỐI CÙNG
        print(f"-> Kiểu dữ liệu của df_to_write: {type(df_to_write)}")
        print(f"-> Số dòng của df_to_write (dùng len()): {len(df_to_write)}")
        print(f"-> Số dòng của df_to_write (dùng .shape): {df_to_write.shape[0]}")
        
        df_to_write.to_csv(DEBUG_CHUNK_OUTPUT_FILE, index=False)
        print(f"-> Đã ghi dữ liệu vào file debug: {DEBUG_CHUNK_OUTPUT_FILE}")

    except Exception as e:
        print(f"[LỖI] Lỗi xảy ra trong quá trình xử lý: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_first_chunk()
