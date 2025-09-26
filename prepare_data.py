import pandas as pd
import numpy as np
import os
import glob

# --- Cấu hình ---
RAW_DATA_DIR = 'raw_data'
PROCESSED_DATA_DIR = 'processed_data'
TRAIN_BENIGN_FILE = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
TEST_LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv')

# Các đặc trưng đã chọn
SELECTED_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Fwd IAT Mean',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
]

def clean_column_names(df):
    """Hàm chuẩn hóa tên cột."""
    cols = df.columns
    new_cols = [col.strip() for col in cols]
    df.columns = new_cols
    return df

def process_file(file_path):
    """Hàm xử lý một file CSV duy nhất."""
    print(f"-> Processing {os.path.basename(file_path)}...")
    
    # Đọc dữ liệu
    df = pd.read_csv(file_path, encoding='latin1')
    df = clean_column_names(df)
    
    # Kiểm tra sự tồn tại của các cột cần thiết
    required_cols = SELECTED_FEATURES + ['Label']
    if not all(col in df.columns for col in required_cols):
        print(f"   [Warning] File {os.path.basename(file_path)} thiếu cột cần thiết. Bỏ qua.")
        return None, None

    # Xử lý dữ liệu bẩn
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Chuyển đổi kiểu dữ liệu để tiết kiệm bộ nhớ
    for col in SELECTED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    # Lấy dữ liệu benign
    df_benign = df[df['Label'] == 'BENIGN'][SELECTED_FEATURES].copy()
    
    # Lấy toàn bộ dữ liệu cho tập test
    df_test = df[SELECTED_FEATURES].copy()
    labels_test = df['Label'].copy()
    
    print(f"   Found {len(df_benign)} benign samples and {len(df_test)} total samples.")
    return df_benign, (df_test, labels_test)

def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    print("--- Bắt đầu quá trình chuẩn bị dữ liệu ---")
    
    # Tạo thư mục output
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Lấy danh sách tất cả các file CSV trong thư mục raw_data
    all_csv_files = glob.glob(os.path.join(RAW_DATA_DIR, '*.csv'))
    
    if not all_csv_files:
        print(f"[Error] Không tìm thấy file CSV nào trong thư mục '{RAW_DATA_DIR}'.")
        return
        
    # Các list để lưu trữ các dataframe con
    benign_dfs = []
    test_dfs = []
    test_labels_dfs = []

    for file_path in all_csv_files:
        benign_part, test_part = process_file(file_path)
        if benign_part is not None and not benign_part.empty:
            benign_dfs.append(benign_part)
        if test_part is not None:
            test_dfs.append(test_part[0])
            test_labels_dfs.append(test_part[1])
    
    if not benign_dfs:
        print("[Error] Không có dữ liệu benign nào được tìm thấy sau khi xử lý.")
        return

    # Gộp tất cả các dataframe lại
    print("\n-> Gộp dữ liệu...")
    final_benign_df = pd.concat(benign_dfs, ignore_index=True)
    final_test_df = pd.concat(test_dfs, ignore_index=True)
    final_test_labels_df = pd.concat(test_labels_dfs, ignore_index=True)

    print(f"   Tổng cộng {len(final_benign_df)} mẫu benign để huấn luyện.")
    print(f"   Tổng cộng {len(final_test_df)} mẫu để kiểm thử.")

    # Lưu kết quả ra file
    print(f"-> Đang lưu file vào thư mục '{PROCESSED_DATA_DIR}'...")
    final_benign_df.to_csv(TRAIN_BENIGN_FILE, index=False)
    final_test_df.to_csv(TEST_FILE, index=False)
    final_test_labels_df.to_csv(TEST_LABELS_FILE, index=False, header=['Label'])

    print("\n--- Quá trình chuẩn bị dữ liệu hoàn tất! ---")

if __name__ == "__main__":
    main()
