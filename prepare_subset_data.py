import pandas as pd
import numpy as np
import os

# --- Cấu hình ---
RAW_DATA_DIR = 'raw_data'
# Thư mục output mới để không ghi đè lên dữ liệu cũ
PROCESSED_DATA_DIR = 'processed_data_subset' 

# Danh sách các file CSV cụ thể mà chúng ta muốn sử dụng
FILES_TO_USE = [
    'Monday-WorkingHours.pcap_ISCX.csv',                      # Chứa 100% traffic BENIGN
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',      # Chứa tấn công DDoS
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'   # Chứa tấn công PortScan
]

# Các đặc trưng đã chọn (giữ nguyên)
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
    
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"   [Warning] File {os.path.basename(file_path)} không tồn tại. Bỏ qua.")
        return None
        
    df = clean_column_names(df)
    
    required_cols = SELECTED_FEATURES + ['Label']
    if not all(col in df.columns for col in required_cols):
        print(f"   [Warning] File {os.path.basename(file_path)} thiếu cột. Bỏ qua.")
        return None

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    for col in SELECTED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    print(f"   Found {len(df)} valid samples.")
    return df

def main():
    """Hàm chính điều phối toàn bộ quá trình."""
    print("--- Bắt đầu quá trình chuẩn bị TẬP DỮ LIỆU CON ---")
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    all_dfs = []
    for filename in FILES_TO_USE:
        file_path = os.path.join(RAW_DATA_DIR, filename)
        processed_df = process_file(file_path)
        if processed_df is not None:
            all_dfs.append(processed_df)
    
    if not all_dfs:
        print("[Error] Không có dữ liệu nào được xử lý.")
        return

    # Gộp tất cả dữ liệu từ các file đã chọn
    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n-> Tổng cộng có {len(final_df)} mẫu trong tập dữ liệu con.")
    
    # Tách dữ liệu ra các file train và test
    print("-> Đang tách dữ liệu thành các file train/test...")
    df_benign = final_df[final_df['Label'] == 'BENIGN']
    
    # File train chỉ chứa dữ liệu benign và các feature đã chọn
    df_train_benign = df_benign[SELECTED_FEATURES].copy()
    
    # File test chứa toàn bộ dữ liệu (cả benign và attack)
    df_test_data = final_df[SELECTED_FEATURES].copy()
    df_test_labels = final_df['Label'].copy()

    # Lưu các file kết quả
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train_benign.csv')
    test_data_path = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
    test_labels_path = os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv')
    
    df_train_benign.to_csv(train_path, index=False)
    df_test_data.to_csv(test_data_path, index=False)
    df_test_labels.to_csv(test_labels_path, index=False, header=['Label'])

    print(f"\n   - Đã lưu {len(df_train_benign)} mẫu train vào '{train_path}'")
    print(f"   - Đã lưu {len(df_test_data)} mẫu test vào '{test_data_path}'")
    print("\n--- Hoàn thành chuẩn bị tập dữ liệu con! ---")

if __name__ == "__main__":
    main()
