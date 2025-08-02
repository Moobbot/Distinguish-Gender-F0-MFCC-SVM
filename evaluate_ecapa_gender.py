import torch
import torchaudio
import sys
sys.path.append('voice-gender-classifier')
from model import ECAPA_gender
from vivos_data_loader import VivosDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

def evaluate_model(dataset, device):
    """Đánh giá mô hình trên dataset"""
    # Load model từ huggingface hub
    model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
    model.eval()
    model.to(device)
    
    # Tạo thư mục tạm để lưu file audio
    os.makedirs('temp_audio', exist_ok=True)
    
    # Khởi tạo lists để lưu kết quả
    true_labels = []
    pred_labels = []
    
    print(f"\nEvaluating on {len(dataset)} samples...")
    
    # Duyệt qua từng mẫu trong dataset
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        waveform = sample['audio']
        sample_rate = sample['sample_rate']
        true_gender = sample['gender']
        
        # Lưu waveform tạm thời thành file
        temp_path = f'temp_audio/temp_{idx}.wav'
        torchaudio.save(temp_path, waveform, sample_rate)
        
        # Dự đoán
        with torch.no_grad():
            pred_gender = model.predict(temp_path, device=device)
        
        # Chuyển đổi nhãn
        true_label = 1 if true_gender == 'm' else 0  # 1 cho nam, 0 cho nữ
        pred_label = 1 if pred_gender == 'male' else 0
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
    
    return true_labels, pred_labels

def plot_confusion_matrix(true_labels, pred_labels, save_path):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Female', 'Male'])
    plt.yticks([0.5, 1.5], ['Female', 'Male'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return cm

def main():
    # Tạo thư mục để lưu kết quả
    os.makedirs('visualize', exist_ok=True)
    
    # Xóa thư mục temp_audio nếu tồn tại
    if os.path.exists('temp_audio'):
        shutil.rmtree('temp_audio')
    
    # Cấu hình device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    vivos_root = "vivos"
    train_dataset = VivosDataset(vivos_root, split='train')
    test_dataset = VivosDataset(vivos_root, split='test')
    
    # Đánh giá trên tập train
    print("\nEvaluating on training set...")
    train_true, train_pred = evaluate_model(train_dataset, device)
    
    # Đánh giá trên tập test
    print("\nEvaluating on test set...")
    test_true, test_pred = evaluate_model(test_dataset, device)
    
    # Vẽ confusion matrix và in kết quả cho tập train
    print("\n=== Training Set Results ===")
    train_cm = plot_confusion_matrix(
        train_true, train_pred, 
        'visualize/confusion_matrix_train.png'
    )
    print("\nTraining Set Classification Report:")
    print(classification_report(train_true, train_pred, 
                              target_names=['Female', 'Male']))
    
    # Vẽ confusion matrix và in kết quả cho tập test
    print("\n=== Test Set Results ===")
    test_cm = plot_confusion_matrix(
        test_true, test_pred, 
        'visualize/confusion_matrix_test.png'
    )
    print("\nTest Set Classification Report:")
    print(classification_report(test_true, test_pred, 
                              target_names=['Female', 'Male']))
    
    # Tính và in các metric bổ sung
    def calculate_metrics(cm):
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1
    
    train_metrics = calculate_metrics(train_cm)
    test_metrics = calculate_metrics(test_cm)
    
    print("\n=== Detailed Metrics ===")
    print("Training Set:")
    print(f"Accuracy: {train_metrics[0]:.4f}")
    print(f"Precision: {train_metrics[1]:.4f}")
    print(f"Recall: {train_metrics[2]:.4f}")
    print(f"F1-score: {train_metrics[3]:.4f}")
    
    print("\nTest Set:")
    print(f"Accuracy: {test_metrics[0]:.4f}")
    print(f"Precision: {test_metrics[1]:.4f}")
    print(f"Recall: {test_metrics[2]:.4f}")
    print(f"F1-score: {test_metrics[3]:.4f}")

if __name__ == "__main__":
    main()
