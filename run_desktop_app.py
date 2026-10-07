"""
Script để chạy ứng dụng Deepfake Detection Desktop
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    try:
        import tkinter
        import torch
        import transformers
        import PIL
        print("Tất cả dependencies đã được cài đặt!")
        return True
    except ImportError as e:
        print(f" Thiếu dependency: {e}")
        print(" Vui lòng cài đặt dependencies:")
        print("pip install torch transformers pillow")
        return False

def check_model():
    """Kiểm tra model có tồn tại không"""
    model_path = Path("deepfake_vs_real_image_detection")
    if not model_path.exists():
        print("Không tìm thấy model đã train!")
        print(" Vui lòng đảm bảo thư mục 'deepfake_vs_real_image_detection' tồn tại")
        return False
    
    # Kiểm tra các file cần thiết
    required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f" Thiếu các file model: {', '.join(missing_files)}")
        return False
    
    print("✅ Model files đã sẵn sàng!")
    return True

def main():
    """Main function để chạy ứng dụng"""
    print(" Khởi động Deepfake Detection Desktop App...")
    
    # Kiểm tra dependencies
    if not check_dependencies():
        input("Nhấn Enter để thoát...")
        return
    
    # Kiểm tra model
    if not check_model():
        input("Nhấn Enter để thoát...")
        return
    
    # Chạy desktop app
    try:
        print(" Đang khởi động ứng dụng desktop...")
        print("Giao diện ứng dụng sẽ hiển thị trong giây lát...")
        print("Đóng cửa sổ ứng dụng để thoát")
        print("-" * 50)
        
        # Import và chạy app
        from deepfake_detector_app import main as run_app
        run_app()
        
    except Exception as e:
        print(f" Lỗi khi chạy ứng dụng: {e}")
        input("Nhấn Enter để thoát...")

if __name__ == "__main__":
    main()
