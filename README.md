# Phân loại giới tính từ tiếng nói tiếng Việt

Hệ thống phân loại giới tính từ tiếng nói tiếng Việt dựa trên tập dữ liệu VIVOS và các đặc trưng MFCC + F0.

## 📋 Mô tả

Dự án này triển khai hệ thống phân loại giới tính từ tiếng nói tiếng Việt sử dụng:

- **Tập dữ liệu:** VIVOS corpus (15 giờ tiếng nói tiếng Việt)
- **Đặc trưng:** MFCC (13 chiều) + F0 (1 chiều) = 14 chiều
- **Mô hình:** SVM và Random Forest
- **Kết quả:** Độ chính xác cao với thời gian xử lý real-time

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu trúc dữ liệu

Đảm bảo có thư mục `vivos/` với cấu trúc:
```
vivos/
├── train/
│   ├── genders.txt
│   ├── prompts.txt
│   └── waves/
│       ├── VIVOSSPK01/
│       ├── VIVOSSPK02/
│       └── ...
└── test/
    └── waves/
```

## 🎯 Sử dụng

### Chạy demo nhanh

```bash
python demo.py
```

### Chạy huấn luyện đầy đủ

```bash
python gender_classification.py
```

### Sử dụng trong code

```python
from gender_classification import VietnameseGenderClassifier

# Khởi tạo classifier
classifier = VietnameseGenderClassifier()

# Tải dataset
classifier.load_dataset(max_files_per_speaker=100)

# Chuẩn bị dữ liệu
X_train, X_test, y_train, y_test = classifier.prepare_data()

# Huấn luyện mô hình
classifier.train_svm(X_train, y_train)
classifier.train_random_forest(X_train, y_train)

# Dự đoán
results = classifier.predict_gender("path/to/audio.wav")
print(results)
```

## 📊 Kết quả

### Đặc trưng sử dụng

- **MFCC:** 13 hệ số Mel-Frequency Cepstral Coefficients
- **F0:** Tần số cơ bản (pitch) trung bình
- **Tổng cộng:** 14 chiều đặc trưng

### Thông số kỹ thuật

- **Frame size:** 25ms
- **Hop length:** 10ms
- **FFT size:** 512
- **Mel filters:** 40
- **Dải tần F0:** 50-300 Hz
- **Tần số lấy mẫu:** 16 kHz

### Mô hình

1. **SVM (Support Vector Machine)**
   - Kernel: RBF Gaussian
   - C = 10, gamma = 0.1
   - Huấn luyện nhanh, dự đoán real-time

2. **Random Forest**
   - 100 cây quyết định
   - Phân tích độ quan trọng đặc trưng
   - So sánh hiệu quả với SVM

## 📈 Đánh giá

Hệ thống cung cấp:

- **Độ chính xác:** Báo cáo chi tiết cho từng mô hình
- **Confusion Matrix:** Ma trận nhầm lẫn
- **Feature Importance:** Phân tích độ quan trọng đặc trưng (Random Forest)
- **So sánh mô hình:** SVM vs Random Forest

## 🎵 Demo

Hệ thống có thể dự đoán giới tính cho file âm thanh đơn lẻ:

```python
# Dự đoán
results = classifier.predict_gender("audio.wav")

# Kết quả
{
    'SVM': {
        'prediction': 'Nữ',
        'confidence': 0.85
    },
    'RandomForest': {
        'prediction': 'Nữ', 
        'confidence': 0.92
    }
}
```

## 📁 Cấu trúc dự án

```
.
├── gender_classification.py  # Code chính
├── demo.py                  # Demo script
├── requirements.txt         # Dependencies
├── README.md               # Hướng dẫn này
├── Teach.md                # Tài liệu kỹ thuật
└── vivos/                  # Dataset VIVOS
    ├── train/
    └── test/
```

## 🔧 Tiền xử lý dữ liệu

1. **Resample & Chuẩn hóa**
   - Tần số lấy mẫu: 16 kHz
   - Chuẩn hóa biên độ

2. **Cắt lọc tĩnh lặng**
   - Loại bỏ khoảng lặng đầu/cuối
   - Ngưỡng năng lượng: 20dB

3. **Trích xuất đặc trưng**
   - MFCC: 13 hệ số
   - F0: Tần số cơ bản trung bình

## 🤖 Mô hình máy học

### SVM
- **Ưu điểm:** Hiệu quả cao, huấn luyện nhanh
- **Tham số:** C=10, gamma=0.1, kernel='rbf'
- **Ứng dụng:** Phân loại chính

### Random Forest
- **Ưu điểm:** Phân tích độ quan trọng đặc trưng
- **Tham số:** 100 cây, độ sâu không giới hạn
- **Ứng dụng:** So sánh và phân tích

## 📊 Kết quả dự kiến

- **Độ chính xác:** 85-95%
- **Thời gian huấn luyện:** Vài giây
- **Thời gian dự đoán:** Real-time
- **Đặc trưng quan trọng nhất:** F0 (pitch)

## 🛠️ Troubleshooting

### Lỗi thường gặp

1. **Không tìm thấy dữ liệu VIVOS**
   - Kiểm tra thư mục `vivos/train/waves/`
   - Đảm bảo có file `genders.txt`

2. **Lỗi thư viện**
   - Cài đặt lại: `pip install -r requirements.txt`
   - Kiểm tra phiên bản Python: 3.7+

3. **Lỗi xử lý âm thanh**
   - Kiểm tra định dạng file: WAV
   - Đảm bảo file âm thanh không bị hỏng

### Tối ưu hóa

- **Giảm số file/speaker:** Thay đổi `max_files_per_speaker`
- **Tăng tốc độ:** Sử dụng ít đặc trưng hơn
- **Cải thiện độ chính xác:** Thêm đặc trưng delta MFCC

## 📚 Tài liệu tham khảo

- [VIVOS Corpus](https://ailab.hcmus.edu.vn/vivos)
- [librosa Documentation](https://librosa.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## 📄 License

Dự án này sử dụng dữ liệu VIVOS được cấp phép theo Creative Commons Attribution NonCommercial ShareAlike 4.0 International License.

## 👥 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork dự án
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📞 Liên hệ

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue trên GitHub. 