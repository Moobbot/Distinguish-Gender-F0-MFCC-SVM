#!/usr/bin/env python3
"""
Phân loại giới tính từ tiếng nói tiếng Việt
Vietnamese Gender Classification from Speech

Dựa trên tập dữ liệu VIVOS và các đặc trưng MFCC + F0
Based on VIVOS dataset and MFCC + F0 features
"""

import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class VietnameseGenderClassifier:
    """
    Hệ thống phân loại giới tính từ tiếng nói tiếng Việt
    """

    def __init__(self, data_path="vivos"):
        """
        Khởi tạo classifier

        Args:
            data_path (str): Đường dẫn đến thư mục dữ liệu VIVOS
        """
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.svm_model = None
        self.rf_model = None
        self.features = []
        self.labels = []

    def load_gender_labels(self):
        """
        Tải nhãn giới tính từ file genders.txt

        Returns:
            dict: Dictionary mapping speaker ID to gender (m/f)
        """
        gender_file = os.path.join(self.data_path, "train", "genders.txt")
        gender_labels = {}

        with open(gender_file, "r", encoding="utf-8") as f:
            for line in f:
                speaker_id, gender = line.strip().split()
                gender_labels[speaker_id] = gender

        return gender_labels

    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Tiền xử lý âm thanh

        Args:
            audio_path (str): Đường dẫn file âm thanh
            target_sr (int): Tần số lấy mẫu mục tiêu

        Returns:
            tuple: (audio_data, sample_rate)
        """
        # Load audio với tần số lấy mẫu mục tiêu
        y, sr = librosa.load(audio_path, sr=target_sr)

        # Chuẩn hóa biên độ
        y = y / np.max(np.abs(y))

        # Cắt lọc tĩnh lặng (silence removal)
        y, _ = librosa.effects.trim(y, top_db=20)

        return y, sr

    def extract_mfcc_features(self, audio_data, sr=16000):
        """
        Trích xuất đặc trưng MFCC

        Args:
            audio_data (np.array): Dữ liệu âm thanh
            sr (int): Tần số lấy mẫu

        Returns:
            np.array: Vector MFCC 13 chiều
        """
        # Tính MFCC với thông số từ Teach.md
        mfcc = librosa.feature.mfcc(
            y=audio_data, sr=sr, n_mfcc=13, n_fft=512, hop_length=160, win_length=400
        )

        # Lấy trung bình theo thời gian
        mfcc_mean = np.mean(mfcc, axis=1)

        return mfcc_mean

    def extract_f0_features(self, audio_data, sr=16000):
        """
        Trích xuất đặc trưng F0 (pitch)

        Args:
            audio_data (np.array): Dữ liệu âm thanh
            sr (int): Tần số lấy mẫu

        Returns:
            float: Giá trị F0 trung bình
        """
        # Tính F0 với thông số từ Teach.md
        f0_series = librosa.yin(audio_data, sr=sr, fmin=50, fmax=300)

        # Lọc bỏ giá trị 0 (khung không có giọng)
        f0_series = f0_series[f0_series > 0]

        # Tính trung bình F0
        f0_mean = np.mean(f0_series) if len(f0_series) > 0 else 0

        return f0_mean

    def extract_energy_features(self, audio_data, sr=16000):
        """
        Trích xuất đặc trưng về năng lượng

        Args:
            audio_data (np.array): Dữ liệu âm thanh
            sr (int): Tần số lấy mẫu

        Returns:
            tuple: (rms_mean, rms_std) - Năng lượng trung bình và độ lệch chuẩn
        """
        # Tính RMS energy theo khung
        frame_length = 2048  # 128ms at 16kHz
        hop_length = 512  # 32ms hop
        rms = librosa.feature.rms(
            y=audio_data, frame_length=frame_length, hop_length=hop_length
        )

        # Tính đặc trưng thống kê
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        return rms_mean, rms_std

    def extract_spectral_features(self, audio_data, sr=16000):
        """
        Trích xuất đặc trưng phổ

        Args:
            audio_data (np.array): Dữ liệu âm thanh
            sr (int): Tần số lấy mẫu

        Returns:
            float: Spectral rolloff - tỷ lệ phổ dải cao/thấp
        """
        # Tính spectral rolloff tại 85% năng lượng
        rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sr, roll_percent=0.85
        )

        # Lấy giá trị trung bình
        rolloff_mean = np.mean(rolloff)

        return rolloff_mean

    def extract_shimmer(self, audio_data, sr=16000):
        """
        Tính độ méo biên độ (shimmer)

        Args:
            audio_data (np.array): Dữ liệu âm thanh
            sr (int): Tần số lấy mẫu

        Returns:
            float: Giá trị shimmer trung bình
        """
        # Chia thành các khung
        frame_length = int(0.03 * sr)  # 30ms frames
        hop_length = int(0.01 * sr)  # 10ms hop
        frames = librosa.util.frame(
            audio_data, frame_length=frame_length, hop_length=hop_length
        )

        # Tính biên độ của mỗi khung
        amplitudes = np.max(np.abs(frames), axis=0)

        # Tính shimmer (độ biến thiên biên độ giữa các khung liên tiếp)
        shimmer = (
            np.mean(np.abs(np.diff(amplitudes)) / amplitudes[:-1])
            if len(amplitudes) > 1
            else 0
        )

        return shimmer

    def extract_features(self, audio_path):
        """
        Trích xuất đặc trưng MFCC + F0 cho một file âm thanh

        Args:
            audio_path (str): Đường dẫn file âm thanh

        Returns:
            np.array: Vector đặc trưng 14 chiều (13 MFCC + 1 F0)
        """
        try:
            # Tiền xử lý âm thanh
            audio_data, sr = self.preprocess_audio(audio_path)

            # Trích xuất MFCC
            mfcc_features = self.extract_mfcc_features(audio_data, sr)

            # Trích xuất F0
            f0_feature = self.extract_f0_features(audio_data, sr)

            # Trích xuất đặc trưng năng lượng
            rms_mean, rms_std = self.extract_energy_features(audio_data, sr)

            # Trích xuất đặc trưng phổ
            rolloff = self.extract_spectral_features(audio_data, sr)

            # Trích xuất shimmer
            shimmer = self.extract_shimmer(audio_data, sr)

            # Kết hợp thành vector 18 chiều (13 MFCC + F0 + 4 đặc trưng mới)
            features = np.concatenate(
                [
                    mfcc_features,  # 13 chiều
                    [f0_feature],  # 1 chiều
                    [rms_mean, rms_std],  # 2 chiều
                    [rolloff],  # 1 chiều
                    [shimmer],  # 1 chiều
                ]
            )

            return features

        except Exception as e:
            print(f"Lỗi khi xử lý file {audio_path}: {e}")
            return None

    def load_dataset(self, max_files_per_speaker=50):
        """
        Tải toàn bộ dataset VIVOS

        Args:
            max_files_per_speaker (int): Số file tối đa mỗi speaker
        """
        print("Đang tải dataset VIVOS...")

        # Tải nhãn giới tính
        gender_labels = self.load_gender_labels()

        # Đường dẫn đến thư mục waves
        waves_path = os.path.join(self.data_path, "train", "waves")

        features_list = []
        labels_list = []

        # Duyệt qua các thư mục speaker
        for speaker_dir in os.listdir(waves_path):
            speaker_path = os.path.join(waves_path, speaker_dir)

            if not os.path.isdir(speaker_path):
                continue

            # Lấy giới tính của speaker
            if speaker_dir not in gender_labels:
                continue

            gender = gender_labels[speaker_dir]
            gender_code = 1 if gender == "f" else 0  # 1: nữ, 0: nam

            # Đếm số file đã xử lý cho speaker này
            files_processed = 0

            # Duyệt qua các file âm thanh
            for audio_file in os.listdir(speaker_path):
                if audio_file.endswith(".wav"):
                    audio_path = os.path.join(speaker_path, audio_file)

                    # Trích xuất đặc trưng
                    features = self.extract_features(audio_path)

                    if features is not None:
                        features_list.append(features)
                        labels_list.append(gender_code)
                        files_processed += 1

                        # Giới hạn số file mỗi speaker
                        if files_processed >= max_files_per_speaker:
                            break

                    # In tiến độ
                    if len(features_list) % 100 == 0:
                        print(f"Đã xử lý {len(features_list)} files...")

        # Chuyển đổi thành numpy arrays
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)

        print(f"Dataset đã tải xong: {len(self.features)} mẫu")
        print(
            f"Phân bố giới tính: Nam={np.sum(self.labels==0)}, Nữ={np.sum(self.labels==1)}"
        )

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Chuẩn bị dữ liệu cho huấn luyện

        Args:
            test_size (float): Tỷ lệ dữ liệu test
            random_state (int): Seed cho random split
        """
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels,
        )

        # Chuẩn hóa đặc trưng
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_svm(self, X_train, y_train):
        """
        Huấn luyện mô hình SVM

        Args:
            X_train (np.array): Đặc trưng huấn luyện
            y_train (np.array): Nhãn huấn luyện
        """
        print("Đang huấn luyện SVM...")

        # SVM với thông số từ Teach.md
        self.svm_model = SVC(kernel="rbf", C=10, gamma=0.1, random_state=42)
        self.svm_model.fit(X_train, y_train)

        print("SVM đã huấn luyện xong!")

    def train_random_forest(self, X_train, y_train):
        """
        Huấn luyện mô hình Random Forest

        Args:
            X_train (np.array): Đặc trưng huấn luyện
            y_train (np.array): Nhãn huấn luyện
        """
        print("Đang huấn luyện Random Forest...")

        # Random Forest với thông số từ Teach.md
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=None, random_state=42
        )
        self.rf_model.fit(X_train, y_train)

        print("Random Forest đã huấn luyện xong!")

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Đánh giá mô hình

        Args:
            model: Mô hình đã huấn luyện
            X_test (np.array): Đặc trưng test
            y_test (np.array): Nhãn test
            model_name (str): Tên mô hình
        """
        # Dự đoán
        y_pred = model.predict(X_test)

        # Tính độ chính xác
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n=== Kết quả {model_name} ===")
        print(f"Độ chính xác: {accuracy:.4f}")

        # Báo cáo chi tiết
        print("\nBáo cáo phân loại:")
        print(classification_report(y_test, y_pred, target_names=["Nam", "Nữ"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        return accuracy, y_pred

    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        """
        Vẽ confusion matrix

        Args:
            y_test (np.array): Nhãn thực tế
            y_pred (np.array): Nhãn dự đoán
            model_name (str): Tên mô hình
        """
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Nam", "Nữ"],
            yticklabels=["Nam", "Nữ"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("Nhãn thực tế")
        plt.xlabel("Nhãn dự đoán")
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.show()

    def analyze_feature_importance(self):
        """
        Phân tích độ quan trọng đặc trưng (chỉ cho Random Forest)
        """
        if self.rf_model is None:
            print("Cần huấn luyện Random Forest trước!")
            return

        # Lấy độ quan trọng đặc trưng
        feature_importance = self.rf_model.feature_importances_

        # Tên đặc trưng
        feature_names = (
            [f"MFCC_{i}" for i in range(13)]  # 13 MFCC
            + ["F0"]  # F0
            + ["RMS_Mean", "RMS_Std"]  # Năng lượng
            + ["Spectral_Rolloff"]  # Đặc trưng phổ
            + ["Shimmer"]  # Độ méo
        )
        # Tạo DataFrame
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importance}
        ).sort_values("Importance", ascending=False)

        print("\n=== Phân tích độ quan trọng đặc trưng ===")
        print(importance_df)

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.title("Độ quan trọng đặc trưng (Random Forest)")
        plt.ylabel("Độ quan trọng")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.show()

    def predict_gender(self, audio_path):
        """
        Dự đoán giới tính cho một file âm thanh

        Args:
            audio_path (str): Đường dẫn file âm thanh

        Returns:
            dict: Kết quả dự đoán
        """
        # Trích xuất đặc trưng
        features = self.extract_features(audio_path)

        if features is None:
            return {"error": "Không thể xử lý file âm thanh"}

        # Chuẩn hóa đặc trưng
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Dự đoán với cả hai mô hình
        results = {}

        if self.svm_model:
            svm_pred = self.svm_model.predict(features_scaled)[0]
            # SVM không có predict_proba mặc định, sử dụng decision_function
            svm_confidence = abs(self.svm_model.decision_function(features_scaled)[0])
            # Chuẩn hóa confidence về khoảng [0, 1]
            svm_confidence = min(svm_confidence / 2.0, 1.0)
            results["SVM"] = {
                "prediction": "Nữ" if svm_pred == 1 else "Nam",
                "confidence": svm_confidence,
            }

        if self.rf_model:
            rf_pred = self.rf_model.predict(features_scaled)[0]
            rf_prob = self.rf_model.predict_proba(features_scaled)[0]
            results["RandomForest"] = {
                "prediction": "Nữ" if rf_pred == 1 else "Nam",
                "confidence": max(rf_prob),
            }

        return results


def main():
    """
    Hàm chính để chạy toàn bộ pipeline
    """
    print("=== Phân loại giới tính từ tiếng nói tiếng Việt ===")
    print("Dựa trên tập dữ liệu VIVOS và đặc trưng MFCC + F0\n")

    # Khởi tạo classifier
    classifier = VietnameseGenderClassifier()

    # Tải dataset
    classifier.load_dataset(max_files_per_speaker=100)

    # Chuẩn bị dữ liệu
    X_train, X_test, y_train, y_test = classifier.prepare_data()

    # Huấn luyện SVM
    classifier.train_svm(X_train, y_train)

    # Huấn luyện Random Forest
    classifier.train_random_forest(X_train, y_train)

    # Đánh giá SVM
    svm_accuracy, svm_pred = classifier.evaluate_model(
        classifier.svm_model, X_test, y_test, "SVM"
    )

    # Đánh giá Random Forest
    rf_accuracy, rf_pred = classifier.evaluate_model(
        classifier.rf_model, X_test, y_test, "Random Forest"
    )

    # Vẽ confusion matrix
    classifier.plot_confusion_matrix(y_test, svm_pred, "SVM")
    classifier.plot_confusion_matrix(y_test, rf_pred, "Random Forest")

    # Phân tích độ quan trọng đặc trưng
    classifier.analyze_feature_importance()

    # So sánh kết quả
    print("\n=== So sánh kết quả ===")
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    if svm_accuracy > rf_accuracy:
        print("SVM cho kết quả tốt hơn!")
    elif rf_accuracy > svm_accuracy:
        print("Random Forest cho kết quả tốt hơn!")
    else:
        print("Cả hai mô hình cho kết quả tương đương!")

    # Demo dự đoán
    print("\n=== Demo dự đoán ===")
    waves_path = os.path.join("vivos", "train", "waves")

    # Tìm một số file để test
    test_files = []
    for speaker_dir in os.listdir(waves_path)[:3]:  # Lấy 3 speaker đầu
        speaker_path = os.path.join(waves_path, speaker_dir)
        if os.path.isdir(speaker_path):
            for audio_file in os.listdir(speaker_path)[:2]:  # Lấy 2 file đầu
                if audio_file.endswith(".wav"):
                    test_files.append(os.path.join(speaker_path, audio_file))
                    break

    for test_file in test_files:
        print(f"\nDự đoán cho file: {os.path.basename(test_file)}")
        results = classifier.predict_gender(test_file)

        if "error" not in results:
            for model_name, result in results.items():
                print(
                    f"{model_name}: {result['prediction']} (confidence: {result['confidence']:.3f})"
                )
        else:
            print(f"Lỗi: {results['error']}")


if __name__ == "__main__":
    main()
