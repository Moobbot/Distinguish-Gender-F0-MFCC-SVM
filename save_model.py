#!/usr/bin/env python3
"""
Script lưu mô hình đã huấn luyện
Save trained models for API deployment
"""

import os
import pickle
import joblib
from gender_classification import VietnameseGenderClassifier

def save_models():
    """
    Huấn luyện và lưu mô hình
    """
    print("=== Lưu mô hình phân loại giới tính ===")
    
    # Khởi tạo classifier
    classifier = VietnameseGenderClassifier()
    
    # Tải dataset
    print("📊 Đang tải dataset...")
    classifier.load_dataset(max_files_per_speaker=100)
    
    # Chuẩn bị dữ liệu
    print("🔧 Đang chuẩn bị dữ liệu...")
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    
    # Huấn luyện mô hình
    print("🤖 Đang huấn luyện SVM...")
    classifier.train_svm(X_train, y_train)
    
    print("🌲 Đang huấn luyện Random Forest...")
    classifier.train_random_forest(X_train, y_train)
    
    # Đánh giá mô hình
    print("📈 Đánh giá mô hình...")
    svm_accuracy, _ = classifier.evaluate_model(
        classifier.svm_model, X_test, y_test, "SVM"
    )
    rf_accuracy, _ = classifier.evaluate_model(
        classifier.rf_model, X_test, y_test, "Random Forest"
    )
    
    # Tạo thư mục models nếu chưa có
    os.makedirs("models", exist_ok=True)
    
    # Lưu mô hình SVM
    print("💾 Đang lưu mô hình SVM...")
    svm_path = "models/svm_model.pkl"
    with open(svm_path, 'wb') as f:
        pickle.dump(classifier.svm_model, f)
    
    # Lưu mô hình Random Forest
    print("💾 Đang lưu mô hình Random Forest...")
    rf_path = "models/random_forest_model.pkl"
    with open(rf_path, 'wb') as f:
        pickle.dump(classifier.rf_model, f)
    
    # Lưu scaler
    print("💾 Đang lưu scaler...")
    scaler_path = "models/scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(classifier.scaler, f)
    
    # Lưu thông tin mô hình
    model_info = {
        'svm_accuracy': svm_accuracy,
        'rf_accuracy': rf_accuracy,
        'feature_names': [f'MFCC_{i}' for i in range(13)] + ['F0'],
        'total_samples': len(classifier.features),
        'male_samples': sum(classifier.labels == 0),
        'female_samples': sum(classifier.labels == 1)
    }
    
    info_path = "models/model_info.pkl"
    with open(info_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print("✅ Đã lưu thành công các mô hình!")
    print(f"📁 Thư mục: models/")
    print(f"   - svm_model.pkl")
    print(f"   - random_forest_model.pkl") 
    print(f"   - scaler.pkl")
    print(f"   - model_info.pkl")
    
    print(f"\n📊 Thông tin mô hình:")
    print(f"   - SVM Accuracy: {svm_accuracy:.4f}")
    print(f"   - Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"   - Tổng số mẫu: {model_info['total_samples']}")
    print(f"   - Nam: {model_info['male_samples']}, Nữ: {model_info['female_samples']}")
    
    return True

if __name__ == "__main__":
    try:
        save_models()
        print("\n🎉 Hoàn thành lưu mô hình!")
    except Exception as e:
        print(f"❌ Lỗi: {e}") 