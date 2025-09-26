import pandas as pd
import numpy as np
import os

# <<< ĐƯỜNG DẪN IMPORT ĐÚNG CHO GES (Greedy Equivalence Search)
from causallearn.search.ScoreBased.GES import ges

from causallearn.utils.GraphUtils import GraphUtils
import time

# --- Cấu hình ---
PROCESSED_DATA_DIR = 'processed_data'
TRAIN_BENIGN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
OUTPUT_DIR = 'results'
GRAPH_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'causal_graph_ges.txt')
GRAPH_IMAGE_FILE = os.path.join(OUTPUT_DIR, 'causal_graph_ges.png')

SAMPLE_SIZE = 100000 

def main():
    """Hàm chính để học đồ thị nhân quả bằng GES."""
    print("--- Bắt đầu Giai đoạn 3: Học Đồ thị Nhân quả với GES ---")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"-> Đang đọc file: {TRAIN_BENIGN_FILE}...")
    try:
        df_benign = pd.read_csv(TRAIN_BENIGN_FILE)
    except FileNotFoundError:
        print(f"[Error] Không tìm thấy file huấn luyện. Hãy chạy script prepare_data.py trước.")
        return

    print(f"   Đọc thành công {len(df_benign)} mẫu benign.")

    if len(df_benign) > SAMPLE_SIZE:
        print(f"-> Lấy mẫu con gồm {SAMPLE_SIZE} dòng để tăng tốc...")
        df_sample = df_benign.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sample = df_benign
        
    # GES hoạt động trực tiếp trên dữ liệu, không cần chuẩn hóa mạnh như NOTEARS
    X = df_sample.to_numpy()
    feature_names = df_sample.columns.tolist()

    print("\n-> Bắt đầu chạy thuật toán GES để khám phá nhân quả...")
    print(f"   Sử dụng CPU.")
    
    start_time = time.time()
    
    # <<< GỌI HÀM GES
    # Hàm ges trả về một dictionary chứa đồ thị ước tính
    # 'bic' là một hàm tính điểm phổ biến cho dữ liệu Gaussian
    Record = ges(X, score_func='local_score_BIC', maxP=None, parameters=None)
    
    # Lấy ma trận kề từ kết quả
    W_est = Record['G'].graph
    
    end_time = time.time()
    print(f"   Hoàn thành GES trong {end_time - start_time:.2f} giây.")

    print("\n-> Đang lưu và trực quan hóa đồ thị...")
    
    np.savetxt(GRAPH_OUTPUT_FILE, W_est, fmt='%d') # Lưu dưới dạng số nguyên 0/1
    print(f"   Đã lưu ma trận kề của đồ thị vào: {GRAPH_OUTPUT_FILE}")
    
    try:
        pydot_graph = GraphUtils.to_pydot(W_est, labels=feature_names)
        pydot_graph.write_png(GRAPH_IMAGE_FILE)
        print(f"   Đã lưu hình ảnh đồ thị vào: {GRAPH_IMAGE_FILE}")
        print("   Lưu ý: Bạn cần chuyển file .png về máy cá nhân để xem.")
    except Exception as e:
        print(f"[Warning] Không thể tạo hình ảnh đồ thị. Lỗi: {e}")
        print("   Hãy chắc chắn rằng bạn đã cài đặt python-graphviz bằng conda.")

    print("\n--- Hoàn thành Giai đoạn 3! ---")

if __name__ == "__main__":
    main()
