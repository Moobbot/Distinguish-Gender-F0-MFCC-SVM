#!/usr/bin/env python3
"""
Demo script cho hệ thống phân loại giới tính từ tiếng nói tiếng Việt
Demo script for Vietnamese gender classification from speech
"""

import os
import sys
from gender_classification import VietnameseGenderClassifier

def quick_demo():
    """
    Demo nhanh với số lượng file ít để test
    """
    print("=== Demo Phân loại giới tính từ tiếng nói tiếng Việt ===")
    print("Sử dụng tập dữ liệu VIVOS với đặc trưng MFCC + F0\n")
    
    # Khởi tạo classifier
    classifier = VietnameseGenderClassifier()
    
    # Kiểm tra xem có dữ liệu VIVOS không
    if not os.path.exists("vivos/train/waves"):
        print("❌ Không tìm thấy dữ liệu VIVOS!")
        print("Vui lòng đảm bảo thư mục 'vivos/train/waves' tồn tại.")
        return
    
    print("✅ Tìm thấy dữ liệu VIVOS")
    
    # Tải dataset với số lượng ít để demo nhanh
    print("\n📊 Đang tải dataset (giới hạn 20 files/speaker)...")
    classifier.load_dataset(max_files_per_speaker=20)
    
    if len(classifier.features) == 0:
        print("❌ Không thể tải được dữ liệu!")
        return
    
    print(f"✅ Đã tải {len(classifier.features)} mẫu")
    print(f"   - Nam: {sum(classifier.labels == 0)}")
    print(f"   - Nữ: {sum(classifier.labels == 1)}")
    
    # Chuẩn bị dữ liệu
    print("\n🔧 Đang chuẩn bị dữ liệu...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(test_size=0.3)
    
    print(f"   - Train: {len(X_train)} mẫu")
    print(f"   - Test: {len(X_test)} mẫu")
    
    # Huấn luyện SVM
    print("\n🤖 Đang huấn luyện SVM...")
    classifier.train_svm(X_train, y_train)
    
    # Huấn luyện Random Forest
    print("\n🌲 Đang huấn luyện Random Forest...")
    classifier.train_random_forest(X_train, y_train)
    
    # Đánh giá
    print("\n📈 Đánh giá kết quả:")
    
    svm_accuracy, svm_pred = classifier.evaluate_model(
        classifier.svm_model, X_test, y_test, "SVM"
    )
    
    rf_accuracy, rf_pred = classifier.evaluate_model(
        classifier.rf_model, X_test, y_test, "Random Forest"
    )
    
    # So sánh
    print(f"\n🏆 So sánh kết quả:")
    print(f"   - SVM: {svm_accuracy:.4f}")
    print(f"   - Random Forest: {rf_accuracy:.4f}")
    
    if svm_accuracy > rf_accuracy:
        print("   🥇 SVM cho kết quả tốt hơn!")
    elif rf_accuracy > svm_accuracy:
        print("   🥇 Random Forest cho kết quả tốt hơn!")
    else:
        print("   🤝 Cả hai mô hình cho kết quả tương đương!")
    
    # Demo dự đoán
    print(f"\n🎯 Demo dự đoán:")
    waves_path = "vivos/train/waves"
    
    # Tìm một số file để test
    test_files = []
    speaker_count = 0
    
    for speaker_dir in os.listdir(waves_path):
        if speaker_count >= 3:  # Chỉ lấy 3 speaker
            break
            
        speaker_path = os.path.join(waves_path, speaker_dir)
        if os.path.isdir(speaker_path):
            # Tìm file âm thanh đầu tiên
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith('.wav'):
                    test_files.append(os.path.join(speaker_path, audio_file))
                    speaker_count += 1
                    break
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n   📁 File {i}: {os.path.basename(test_file)}")
        results = classifier.predict_gender(test_file)
        
        if 'error' not in results:
            for model_name, result in results.items():
                gender_emoji = "👩" if result['prediction'] == 'Nữ' else "👨"
                print(f"      {model_name}: {gender_emoji} {result['prediction']} (confidence: {result['confidence']:.3f})")
        else:
            print(f"      ❌ Lỗi: {results['error']}")
    
    print(f"\n✅ Demo hoàn thành!")
    print(f"📊 Kết quả tốt nhất: {max(svm_accuracy, rf_accuracy):.4f}")

def test_single_file():
    """
    Test với một file âm thanh cụ thể
    """
    print("\n=== Test với file đơn lẻ ===")
    
    # Tìm một file âm thanh để test
    waves_path = "vivos/train/waves"
    test_file = None
    
    for speaker_dir in os.listdir(waves_path):
        speaker_path = os.path.join(waves_path, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith('.wav'):
                    test_file = os.path.join(speaker_path, audio_file)
                    break
            if test_file:
                break
    
    if test_file:
        print(f"🎵 Test file: {os.path.basename(test_file)}")
        
        # Khởi tạo classifier
        classifier = VietnameseGenderClassifier()
        
        # Trích xuất đặc trưng
        features = classifier.extract_features(test_file)
        
        if features is not None:
            print(f"✅ Trích xuất được {len(features)} đặc trưng")
            print(f"   - MFCC: {features[:13]}")
            print(f"   - F0: {features[13]:.2f}")
        else:
            print("❌ Không thể trích xuất đặc trưng")
    else:
        print("❌ Không tìm thấy file âm thanh để test")

if __name__ == "__main__":
    try:
        quick_demo()
        test_single_file()
    except KeyboardInterrupt:
        print("\n⏹️ Demo bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("Vui lòng kiểm tra:")
        print("1. Dữ liệu VIVOS có tồn tại không")
        print("2. Các thư viện đã được cài đặt chưa")
        print("3. File gender_classification.py có tồn tại không") 