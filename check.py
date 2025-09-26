import torch

print("--- PyTorch and CUDA Verification ---")

# 1. Kiểm tra phiên bản PyTorch
print(f"PyTorch Version: {torch.__version__}")

# 2. Kiểm tra khả năng truy cập CUDA
is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda_available}")

# 3. Nếu CUDA có sẵn, in thông tin chi tiết về các GPU
if is_cuda_available:
    # Lấy số lượng GPU
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    
    # In thông tin cho từng GPU
    for i in range(gpu_count):
        print(f"\n--- GPU {i} ---")
        print(f"Device Name: {torch.cuda.get_device_name(i)}")
        
        # Kiểm tra khả năng tính toán của GPU
        capability = torch.cuda.get_device_capability(i)
        print(f"Compute Capability: {capability[0]}.{capability[1]}")
        
        # Lấy tổng bộ nhớ GPU
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3) # Convert bytes to GB
        print(f"Total Memory: {total_mem:.2f} GB")
else:
    print("Warning: PyTorch was installed, but it cannot find CUDA.")
    print("Please check your NVIDIA driver and CUDA toolkit installation.")

print("\n--- Verification Complete ---")
