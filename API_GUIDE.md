# Hướng dẫn sử dụng API Phân loại giới tính từ tiếng nói tiếng Việt

## 🚀 Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
# Kích hoạt môi trường ảo
.venv/Scripts/activate

# Cài đặt thư viện API
pip install fastapi uvicorn python-multipart requests
```

### 2. Lưu mô hình

```bash
# Huấn luyện và lưu mô hình
python save_model.py
```

### 3. Chạy API server

```bash
# Chạy API server
python api.py
```

API sẽ chạy tại: http://localhost:8000

## 📋 Các endpoint

### 1. **GET /** - Thông tin API
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Vietnamese Gender Classification API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "/": "API info",
    "/predict": "Upload audio file for gender classification",
    "/health": "Health check",
    "/model-info": "Model information"
  }
}
```

### 2. **GET /health** - Kiểm tra trạng thái
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["svm", "rf", "scaler", "info"]
}
```

### 3. **GET /model-info** - Thông tin mô hình
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_info": {
    "svm_accuracy": 1.0,
    "rf_accuracy": 0.9978,
    "feature_names": ["MFCC_0", "MFCC_1", ..., "F0"],
    "total_samples": 920,
    "male_samples": 440,
    "female_samples": 480
  },
  "available_models": ["SVM", "RandomForest"],
  "features": ["MFCC_0", "MFCC_1", ..., "F0"]
}
```

### 4. **POST /predict** - Dự đoán giới tính

#### Sử dụng curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio_file.wav"
```

#### Sử dụng Python requests:
```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('audio_file.wav', 'rb')}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

#### Response:
```json
{
  "success": true,
  "prediction": {
    "success": true,
    "predictions": {
      "SVM": {
        "prediction": "Nữ",
        "confidence": 0.85,
        "gender_code": 1
      },
      "RandomForest": {
        "prediction": "Nữ", 
        "confidence": 0.92,
        "gender_code": 1
      }
    },
    "final_prediction": "Nữ",
    "final_confidence": 0.92,
    "features": {
      "mfcc": [-349.08, 72.90, -32.51, ...],
      "f0": 209.70,
      "feature_vector": [-349.08, 72.90, ..., 209.70]
    },
    "model_info": {
      "svm_accuracy": 1.0,
      "rf_accuracy": 0.9978,
      "total_samples": 920
    }
  }
}
```

### 5. **POST /predict-url** - Dự đoán từ URL
```bash
curl -X POST "http://localhost:8000/predict-url" \
     -H "Content-Type: application/json" \
     -d '{"audio_url": "https://example.com/audio.wav"}'
```

## 🎵 Định dạng file hỗ trợ

- **WAV** (.wav) - Khuyến nghị
- **MP3** (.mp3)
- **FLAC** (.flac)
- **M4A** (.m4a)
- **OGG** (.ogg)

## 📊 Giải thích kết quả

### Cấu trúc response:

| Trường | Mô tả |
|--------|-------|
| `success` | Trạng thái thành công |
| `predictions.SVM` | Kết quả từ mô hình SVM |
| `predictions.RandomForest` | Kết quả từ mô hình Random Forest |
| `final_prediction` | Dự đoán cuối cùng (Nam/Nữ) |
| `final_confidence` | Độ tin cậy cao nhất |
| `features.mfcc` | 13 hệ số MFCC |
| `features.f0` | Tần số cơ bản |
| `model_info` | Thông tin mô hình |

### Confidence levels:
- **0.0 - 0.5**: Độ tin cậy thấp
- **0.5 - 0.8**: Độ tin cậy trung bình  
- **0.8 - 1.0**: Độ tin cậy cao

## 🧪 Test API

### Chạy test tự động:
```bash
python test_api.py
```

### Test thủ công:
```bash
# 1. Kiểm tra health
curl http://localhost:8000/health

# 2. Xem thông tin mô hình
curl http://localhost:8000/model-info

# 3. Test dự đoán
curl -X POST "http://localhost:8000/predict" \
     -F "file=@vivos/train/waves/VIVOSSPK01/VIVOSSPK01_R001.wav"
```

## 🌐 Swagger UI

Truy cập: http://localhost:8000/docs

- Giao diện web để test API
- Tài liệu tự động
- Thử nghiệm trực tiếp

## 📱 Ví dụ sử dụng

### Python client:
```python
import requests
import json

def predict_gender(audio_file_path):
    """Dự đoán giới tính từ file âm thanh"""
    
    url = "http://localhost:8000/predict"
    
    with open(audio_file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            prediction = result['prediction']
            print(f"Giới tính: {prediction['final_prediction']}")
            print(f"Độ tin cậy: {prediction['final_confidence']:.3f}")
            print(f"SVM: {prediction['predictions']['SVM']['prediction']}")
            print(f"Random Forest: {prediction['predictions']['RandomForest']['prediction']}")
        else:
            print(f"Lỗi: {result['error']}")
    else:
        print(f"HTTP Error: {response.status_code}")

# Sử dụng
predict_gender("audio_file.wav")
```

### JavaScript client:
```javascript
async function predictGender(audioFile) {
    const formData = new FormData();
    formData.append('file', audioFile);
    
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            const prediction = result.prediction;
            console.log(`Giới tính: ${prediction.final_prediction}`);
            console.log(`Độ tin cậy: ${prediction.final_confidence}`);
        } else {
            console.error(`Lỗi: ${result.error}`);
        }
    } catch (error) {
        console.error('Lỗi kết nối:', error);
    }
}
```

## ⚠️ Lưu ý quan trọng

1. **File âm thanh**: Nên sử dụng định dạng WAV, chất lượng tốt
2. **Độ dài**: File nên có độ dài 2-10 giây
3. **Chất lượng**: Âm thanh rõ ràng, ít nhiễu
4. **Giới hạn**: File size < 10MB
5. **Rate limit**: Không có giới hạn cứng

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **"Models not loaded"**
   - Chạy `python save_model.py` trước
   - Kiểm tra thư mục `models/`

2. **"Unsupported file format"**
   - Sử dụng định dạng được hỗ trợ
   - Kiểm tra extension file

3. **"Cannot extract features"**
   - File âm thanh bị hỏng
   - Thử file khác

4. **Connection refused**
   - API server chưa chạy
   - Chạy `python api.py`

## 📈 Performance

- **Thời gian xử lý**: ~1-3 giây/file
- **Độ chính xác**: 99.78% - 100%
- **Memory usage**: ~200MB
- **Concurrent requests**: Hỗ trợ nhiều request đồng thời

## 🔒 Security

- API chạy localhost (không expose ra internet)
- File tạm được xóa sau khi xử lý
- Không lưu trữ file upload
- Validation đầy đủ input

---

**Phiên bản API**: 1.0.0  
**Dữ liệu**: VIVOS Corpus  
**Mô hình**: SVM + Random Forest  
**Đặc trưng**: MFCC + F0 