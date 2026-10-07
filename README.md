# Deepfake vs Real Faces Detection using Vision Transformer (ViT)

Dự án phát hiện deepfake sử dụng Vision Transformer (ViT) để phân loại ảnh thật và ảnh giả.
deepfake là gì 

## Cấu trúc dự án

```
├── config.py                 # Cấu hình và tham số
├── data_processor.py         # Xử lý dữ liệu
├── model.py                  # Model ViT và training
├── visualization.py          # Hiển thị kết quả và biểu đồ
├── main.py                   # Script chính chạy toàn bộ pipeline
├── requirements.txt          # Thư viện cần thiết
└── README.md                 # Hướng dẫn sử dụng
```

## Yêu cầu hệ thống

- Python 3.11.9
- CUDA (tùy chọn, để tăng tốc GPU)
- RAM: ít nhất 8GB (khuyến nghị 16GB)

## Cài đặt

2. **Tạo môi trường ảo:**

```bash
python -m venv venv
```

3. **Kích hoạt môi trường:**

```bash
# Windows
venv\Scripts\activate
```

4. **Cài đặt thư viện:**

```bash
pip install -r requirements.txt
python run_desktop_app.py

```

## Cấu trúc dữ liệu

Đảm bảo dữ liệu của bạn có cấu trúc như sau:

```
Dataset/
├── Train/
│   ├── Real/
│   │   └── [ảnh thật]
│   └── Fake/
│       └── [ảnh giả]
├── Test/
│   ├── Real/
│   │   └── [ảnh thật]
│   └── Fake/
│       └── [ảnh giả]
└── Validation/
    ├── Real/
    │   └── [ảnh thật]
    └── Fake/
        └── [ảnh giả]
```



## Cấu hình

Chỉnh sửa file `config.py` để thay đổi các tham số:

- `DATASET_PATH`: Đường dẫn đến dataset
- `NUM_EPOCHS`: Số epoch training
- `BATCH_SIZE`: Batch size
- `LEARNING_RATE`: Learning rate
- `MODEL_NAME`: Tên model pre-trained



## 📋 Yêu cầu hệ thống

- Python 3.8+
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- GPU: Không bắt buộc nhưng sẽ tăng tốc độ xử lý
- Model đã train: Thư mục `deepfake_vs_real_image_detection`

## 🎯 Cách sử dụng

1. **Chọn ảnh**:

   - Click nút "📁 Chọn ảnh từ máy tính"
   - Duyệt và chọn file ảnh (JPG, PNG, JPEG, etc.)

2. **Phân tích**:

   - Click nút "🔍 Phân tích ảnh"
   - Chờ mô hình xử lý (thường mất 2-5 giây)

3. **Xem kết quả**:

   - Kết quả: Real (Thật) hoặc Fake (Giả)
   - Độ tin cậy: Phần trăm confidence score
   - Xác suất chi tiết: Progress bars cho từng class

4. **Lưu kết quả**:
   - Click "💾 Lưu kết quả" để xuất ra file text
