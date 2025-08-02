# Báo cáo kết quả: Đánh giá model ECAPA trên dữ liệu VIVOS

## 📊 Tổng quan kết quả

### 🎯 Hiệu suất hệ thống

- **Tổng số mẫu train:** 11,660 files
- **Tổng số mẫu test:** 760 files
- **Phân bố train:** 6,659 Nữ, 5,001 Nam
- **Phân bố test:** 280 Nữ, 480 Nam
- **Thời gian dự đoán:** ~7.12 mẫu/giây trên CPU

## 🤖 Kết quả mô hình

### Tập huấn luyện (Training Set)

- **Accuracy:** 95.51%
- **Precision (Male):** 99.98%
- **Recall (Male):** 89.54%
- **F1-score (Male):** 94.47%
- **Confusion Matrix:** Được lưu trong `visualize/confusion_matrix_train.png`

Chi tiết theo từng lớp:
```
              precision    recall  f1-score   support
      Female       0.93      1.00      0.96      6659
        Male       1.00      0.90      0.94      5001
    accuracy                           0.96     11660
   macro avg       0.96      0.95      0.95     11660
weighted avg       0.96      0.96      0.95     11660
```

### Tập kiểm thử (Test Set)

- **Accuracy:** 96.32%
- **Precision (Male):** 100%
- **Recall (Male):** 94.17%
- **F1-score (Male):** 97%
- **Confusion Matrix:** Được lưu trong `visualize/confusion_matrix_test.png`

Chi tiết theo từng lớp:
```
              precision    recall  f1-score   support
      Female       0.91      1.00      0.95       280
        Male       1.00      0.94      0.97       480
    accuracy                           0.96       760
   macro avg       0.95      0.97      0.96       760
weighted avg       0.97      0.96      0.96       760
```

## ✅ Phân tích kết quả

### Điểm mạnh

1. ✅ **Độ chính xác cao:** >95% trên cả tập train và test
2. ✅ **Không overfitting:** Performance trên test set thậm chí còn tốt hơn train set
3. ✅ **Precision hoàn hảo cho Male:** 100% trên test set - khi dự đoán là nam thì luôn đúng
4. ✅ **Recall tốt:** >90% cho cả nam và nữ

### Đặc điểm

1. **Thiên lệch nhẹ về giới tính:**
   - Nhận diện giọng nữ tốt hơn (recall 100%)
   - Một số giọng nam bị nhận diện nhầm thành nữ (recall 90-94%)

2. **Ổn định giữa train và test:**
   - Train accuracy: 95.51%
   - Test accuracy: 96.32%
   - Chênh lệch chỉ 0.81%

### Khuyến nghị

1. **Cân bằng dữ liệu:** Số lượng mẫu nam/nữ chưa cân bằng trong cả train và test
2. **Tối ưu tốc độ:** 7.12 mẫu/giây trên CPU có thể cải thiện bằng cách:
   - Sử dụng GPU
   - Batch processing
   - Tối ưu preprocessing
3. **Cải thiện recall cho Male:** Có thể điều chỉnh ngưỡng phân loại để cân bằng hơn giữa nam và nữ

## 📊 Thống kê tổng hợp

| Chỉ số | Train | Test |
|--------|-------|------|
| Tổng số mẫu | 11,660 | 760 |
| Số mẫu Nam | 5,001 | 480 |
| Số mẫu Nữ | 6,659 | 280 |
| Accuracy | 95.51% | 96.32% |
| Precision (Male) | 99.98% | 100% |
| Recall (Male) | 89.54% | 94.17% |
| F1-score (Male) | 94.47% | 97% |

---

**Ngày tạo báo cáo:** August 2, 2025  
**Model:** ECAPA-TDNN from "JaesungHuh/voice-gender-classifier"  
**Dữ liệu:** VIVOS Vietnamese Speech Corpus
