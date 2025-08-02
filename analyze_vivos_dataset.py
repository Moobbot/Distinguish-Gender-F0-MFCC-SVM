import os
import numpy as np
import matplotlib.pyplot as plt
from vivos_data_loader import VivosDataset
import torchaudio
import librosa

def analyze_audio_duration(dataset):
    """Phân tích độ dài các file audio"""
    durations = []
    for sample in dataset.data_list:
        waveform, sample_rate = torchaudio.load(sample['audio_path'])
        duration = waveform.shape[1] / sample_rate
        durations.append(duration)
    return durations

def analyze_text_length(dataset):
    """Phân tích độ dài của các đoạn text"""
    text_lengths = [len(sample['text'].split()) for sample in dataset.data_list]
    return text_lengths

def plot_stats(train_dataset, test_dataset):
    """Vẽ các biểu đồ thống kê"""
    
    # 1. Gender Distribution
    plt.figure(figsize=(12, 6))
    
    # Train set
    plt.subplot(1, 2, 1)
    train_gender_counts = train_dataset.get_gender_stats()
    plt.bar(train_gender_counts.keys(), train_gender_counts.values())
    plt.title('Gender Distribution - Train Set')
    plt.ylabel('Number of samples')
    
    # Test set
    plt.subplot(1, 2, 2)
    test_gender_counts = test_dataset.get_gender_stats()
    plt.bar(test_gender_counts.keys(), test_gender_counts.values())
    plt.title('Gender Distribution - Test Set')
    
    plt.tight_layout()
    plt.savefig('visualize/gender_distribution.png')
    plt.close()

    # 2. Audio Duration Distribution
    train_durations = analyze_audio_duration(train_dataset)
    plt.figure(figsize=(10, 6))
    plt.hist(train_durations, bins=50)
    plt.title('Audio Duration Distribution (Train Set)')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.savefig('visualize/audio_duration_histogram.png')
    plt.close()

    # 3. Text Length Distribution
    train_text_lengths = analyze_text_length(train_dataset)
    plt.figure(figsize=(10, 6))
    plt.hist(train_text_lengths, bins=30)
    plt.title('Text Length Distribution (Train Set)')
    plt.xlabel('Number of words')
    plt.ylabel('Count')
    plt.savefig('visualize/text_length_histogram.png')
    plt.close()

    # 4. Top Speakers
    speaker_counts, _ = train_dataset.get_speaker_stats()
    top_speakers = dict(sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_speakers.keys(), top_speakers.values())
    plt.title('Top 10 Speakers by Number of Samples')
    plt.xticks(rotation=45)
    plt.ylabel('Number of samples')
    plt.tight_layout()
    plt.savefig('visualize/top_speakers_count.png')
    plt.close()

def print_detailed_stats(train_dataset, test_dataset):
    """In các thống kê chi tiết"""
    print("\n=== DETAILED DATASET STATISTICS ===")
    
    # Train set stats
    train_gender_counts = train_dataset.get_gender_stats()
    train_speaker_counts, train_speaker_genders = train_dataset.get_speaker_stats()
    
    print("\nTRAIN SET:")
    print(f"Total samples: {len(train_dataset)}")
    print(f"Total speakers: {len(train_speaker_counts)}")
    print(f"Gender distribution: {train_gender_counts}")
    print("\nAverage samples per speaker:", 
          sum(train_speaker_counts.values()) / len(train_speaker_counts))
    
    # Test set stats
    test_gender_counts = test_dataset.get_gender_stats()
    test_speaker_counts, test_speaker_genders = test_dataset.get_speaker_stats()
    
    print("\nTEST SET:")
    print(f"Total samples: {len(test_dataset)}")
    print(f"Total speakers: {len(test_speaker_counts)}")
    print(f"Gender distribution: {test_gender_counts}")
    print("\nAverage samples per speaker:", 
          sum(test_speaker_counts.values()) / len(test_speaker_counts))
    
    # Audio duration stats
    train_durations = analyze_audio_duration(train_dataset)
    print("\nAUDIO STATISTICS (Train Set):")
    print(f"Average duration: {np.mean(train_durations):.2f} seconds")
    print(f"Min duration: {np.min(train_durations):.2f} seconds")
    print(f"Max duration: {np.max(train_durations):.2f} seconds")
    
    # Text length stats
    train_text_lengths = analyze_text_length(train_dataset)
    print("\nTEXT STATISTICS (Train Set):")
    print(f"Average words per sample: {np.mean(train_text_lengths):.2f}")
    print(f"Min words: {np.min(train_text_lengths)}")
    print(f"Max words: {np.max(train_text_lengths)}")

if __name__ == "__main__":
    # Đường dẫn đến dataset
    vivos_root = "vivos"
    
    # Tạo dataset objects
    train_dataset = VivosDataset(vivos_root, split='train')
    test_dataset = VivosDataset(vivos_root, split='test')
    
    # Tạo thư mục visualize nếu chưa tồn tại
    os.makedirs('visualize', exist_ok=True)
    
    # Vẽ các biểu đồ thống kê
    plot_stats(train_dataset, test_dataset)
    
    # In thống kê chi tiết
    print_detailed_stats(train_dataset, test_dataset)
