# Phân loại giới tính từ tiếng nói tiếng Việt

## 2.2 Thiết kế & triển khai

### 2.2.1 Dữ liệu và công cụ

#### Tập dữ liệu VIVOS

Toàn bộ dữ liệu huấn luyện và kiểm thử được lấy từ **VIVOS corpus** – một bộ dữ liệu tiếng Việt thu âm do AILAB, Đại học Khoa học Tự nhiên TP.HCM phát hành tự do trên Kaggle.

**Đặc điểm chính:**

- **Thời lượng:** Khoảng 15 giờ tiếng nói (giọng đọc)
- **Phân chia:** 16 tiếng huấn luyện và 0.75 tiếng kiểm thử
- **Người nói:** 65 người (bao gồm cả nam và nữ)
- **Môi trường:** Thu âm trong môi trường yên tĩnh, sử dụng microphone chất lượng cao
- **Định dạng:** Tệp WAV ngắn (mỗi tệp chứa một câu đọc)
- **Nhãn:** Có nhãn giới tính tương ứng cho từng tệp

**Cấu trúc dữ liệu:**

- **Tập train:** Dùng để huấn luyện mô hình
- **Tập test:** ~45 phút âm thanh, dùng để đánh giá
- **Validation:** Có thể tách từ tập train để tinh chỉnh tham số

#### Môi trường và công cụ

Việc triển khai được thực hiện bằng **Python 3.11** với các thư viện chính:

**Thư viện xử lý âm thanh:**

- `librosa` - Xử lý âm thanh và trích xuất đặc trưng (MFCC, F0)
- `numpy` và `scipy` - Xử lý tín hiệu số

**Thư viện máy học:**

- `scikit-learn` - Cài đặt các mô hình máy học (SVM, Random Forest) và công cụ đánh giá

**Thư viện trực quan hóa:**

- `matplotlib/seaborn` - Vẽ biểu đồ kết quả (phổ, MFCC, phân bố F0, confusion matrix)

**Thư viện thu âm:**

- `pyaudio` hoặc `sounddevice` - Thu âm từ microphone (cho real-time demo)

**Thư viện học sâu:**

- `torch` - Tải và chạy mô hình học sâu pre-trained (ECAPA-TDNN)

**Đặc điểm triển khai:**

- Mã nguồn thuần Python
- Chạy được trên môi trường CPU
- Thời gian xử lý real-time (phân loại nhanh hơn độ dài tín hiệu)

### 2.2.2 Tiền xử lý dữ liệu

Dữ liệu âm thanh từ VIVOS có định dạng **WAV (16-bit PCM, mono)**. Trước khi trích xuất đặc trưng, mỗi tệp âm thanh được đưa qua các bước tiền xử lý sau:

#### 1. Resample & Chuẩn hóa

- **Tần số lấy mẫu:** Đảm bảo tất cả âm thanh ở cùng tần số lấy mẫu (16 kHz)
- **VIVOS gốc:** Thu ở 16 kHz nên không cần resample
- **Chuẩn hóa biên độ:** Chia cho giá trị tối đa để tránh khác biệt do âm lượng

#### 2. Cắt lọc tĩnh lặng

- **Phương pháp:** Cắt bớt khoảng lặng dựa trên ngưỡng năng lượng (energy threshold)
- **Mục đích:** Chỉ giữ phần có tiếng nói
- **Lợi ích:** Đặc trưng tập trung vào phần thoại, tránh gây nhiễu khi tính MFCC và F0

#### 3. Phân đoạn (nếu cần)

- **Độ dài hiện tại:** Mỗi tệp VIVOS là một câu ngắn (trung bình 2-3 giây)
- **Xử lý:** Độ dài này phù hợp để xử lý nguyên đoạn
- **Quyết định:** Giữ mỗi câu làm một mẫu cho mô hình

Sau tiền xử lý sẽ thu được tập các đoạn âm thanh sẵn sàng để trích xuất đặc trưng.

### 2.2.3 Trích xuất đặc trưng MFCC và F0

Từ mỗi đoạn âm thanh (sau tiền xử lý), chúng tôi trích xuất các đặc trưng sau để tạo vector đặc trưng đầu vào cho mô hình phân loại:

#### 1. MFCC (Mel-Frequency Cepstral Coefficients)

**Thông số kỹ thuật:**

- **Số hệ số:** 13 hệ số MFCC (bao gồm hệ số 0 năng lượng)
- **Frame size:** 25ms
- **Hop length:** 10ms
- **FFT size:** 512
- **Mel filters:** 40 filter
- **Dải tần:** 0-8000 Hz (nửa phổ do tín hiệu 16kHz)

**Xử lý:**

- Tính MFCC cho từng frame
- Lấy trung bình qua toàn bộ các frame của đoạn
- Thu được vector MFCC 13 chiều đại diện cho đoạn

**Code minh họa:**

```python
import librosa

y, sr = librosa.load(file_path, sr=16000)
mfcc = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=13, n_fft=512, 
    hop_length=160, win_length=400
)
mfcc_mean = mfcc.mean(axis=1)  # trung bình theo cột (theo thời gian)
```

**Lưu ý:** Đã thử nghiệm delta và delta-delta (39 chiều) nhưng không cải thiện đáng kể kết quả.

#### 2. F0 (Pitch - Tần số cơ bản)

**Thông số kỹ thuật:**

- **Phương pháp:** Sử dụng hàm `librosa.yin`
- **Dải tần F0:** fmin=50 Hz, fmax=300 Hz (phù hợp giọng người)
- **Xử lý:** Bỏ qua các khung voiceless có F0=0

**Code minh họa:**

```python
f0_series = librosa.yin(y, sr=sr, fmin=50, fmax=300)
f0_series = f0_series[f0_series > 0]  # lọc bỏ giá trị 0
f0_mean = f0_series.mean() if len(f0_series) > 0 else 0
```

**Xử lý ngoại lệ:** Nếu đoạn toàn là âm vô thanh, cho F0_mean = 0

#### 3. Vector đặc trưng cuối cùng

**Kích thước:** 14 chiều (13 MFCC + 1 F0_mean)

**Các đặc trưng khác đã thử nghiệm:**

- Độ năng lượng trung bình
- Độ lệch chuẩn năng lượng
- Độ méo tiếng (jitter, shimmer)
- Tỷ lệ phổ dải cao/dải thấp

**Kết quả:** Không tăng nhiều hiệu quả khi đã có MFCC+F0

**Dữ liệu đầu ra:**

- **Tập huấn luyện:** ~11,660 đoạn
- **Tập test:** ~760 đoạn
- **Mỗi mẫu:** Vector 14 chiều + nhãn giới tính (Nam/Nữ)

### 2.2.4 Lựa chọn mô hình phân loại

Với vector đặc trưng 14 chiều, thực hiện thử nghiệm hai mô hình truyền thống:

#### 1. SVM (Support Vector Machine)

**Lý do chọn:** Hiệu quả cao trong các nghiên cứu trước

**Cấu hình:**

- **Kernel:** RBF Gaussian (phi tuyến)
- **Tham số:** C = 10, gamma = 0.1
- **Tối ưu:** Tìm kiếm nhanh trên tập validation

**Ưu điểm:**

- Huấn luyện nhanh (vài giây trên ~11k mẫu)
- Dự đoán nhanh (trong tích tắc)
- Hiệu quả cao với không gian đặc trưng nhỏ

**Code huấn luyện:**

```python
from sklearn.svm import SVC

model_svm = SVC(kernel='rbf', C=10, gamma=0.1)
model_svm.fit(X_train, y_train)  # X_train: mảng Nx14, y_train: nhãn 0/1
```

**Kết quả:** Học được siêu phẳng tối ưu trong không gian 14D để phân biệt hai lớp.

#### 2. Random Forest

**Lý do chọn:** So sánh hiệu quả và phân tích độ quan trọng đặc trưng

**Cấu hình:**

- **Số cây:** 100 cây
- **Độ sâu:** Không giới hạn (cho đến khi lá thuần nhất)

**Đặc điểm:**

- Huấn luyện rất nhanh (vài giây)
- Hoạt động tốt với dữ liệu nhiều đặc trưng rời rạc
- Có thể kém SVM một chút về độ chính xác trong không gian nhỏ

**Lợi ích:**

- Phân tích độ quan trọng đặc trưng
- Dự kiến F0 đóng vai trò lớn nhất so với các hệ số MFCC
