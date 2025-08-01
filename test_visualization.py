import matplotlib.pyplot as plt
import numpy as np
import os
from vivos_data_loader import VivosDataset

def create_simple_visualizations(dataset, output_dir="visualy"):
    """Tạo các visualization đơn giản và lưu vào folder"""
    # Tạo folder nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating simple visualizations in folder: {output_dir}")
    
    # 1. Audio duration histogram
    print("1. Creating audio duration histogram...")
    durations = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        duration = sample['audio'].shape[1] / sample['sample_rate']
        durations.append(duration)
    
    durations = np.array(durations)
    
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Audio Durations')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "audio_duration_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Audio duration histogram saved to: {os.path.join(output_dir, 'audio_duration_histogram.png')}")
    
    # 2. Text length histogram
    print("2. Creating text length histogram...")
    text_lengths = []
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        text_length = len(sample['text'].split())
        text_lengths.append(text_length)
    
    text_lengths = np.array(text_lengths)
    
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Lengths')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "text_length_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Text length histogram saved to: {os.path.join(output_dir, 'text_length_histogram.png')}")
    
    # 3. Sample audio waveforms
    print("3. Creating sample audio waveforms...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        
        plt.figure(figsize=(12, 4))
        plt.plot(sample['audio'].squeeze().numpy())
        plt.title(f"Audio Waveform: {sample['audio_id']}")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"waveform_{sample['audio_id']}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Waveform saved to: {os.path.join(output_dir, f'waveform_{sample['audio_id']}.png')}")
    
    # 4. Gender distribution pie chart
    print("4. Creating gender distribution chart...")
    gender_counts = dataset.get_gender_stats()
    
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%')
    plt.title('Gender Distribution in VIVOS Dataset')
    plt.savefig(os.path.join(output_dir, "gender_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gender distribution chart saved to: {os.path.join(output_dir, 'gender_distribution.png')}")
    
    # 5. Speaker count bar chart
    print("5. Creating speaker count chart...")
    speaker_counts, _ = dataset.get_speaker_stats()
    
    # Lấy top 10 speakers
    top_speakers = sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    speakers, counts = zip(*top_speakers)
    
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(speakers)), counts)
    plt.xlabel('Speaker ID')
    plt.ylabel('Number of Samples')
    plt.title('Top 10 Speakers by Sample Count')
    plt.xticks(range(len(speakers)), speakers, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_speakers_count.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top speakers chart saved to: {os.path.join(output_dir, 'top_speakers_count.png')}")
    
    print(f"\nAll visualizations saved to folder: {output_dir}")
    print("Files created:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == "__main__":
    print("=== CREATING SIMPLE VISUALIZATIONS ===")
    
    # Load dataset
    dataset = VivosDataset("vivos", split='train')
    
    # Tạo tất cả visualizations
    create_simple_visualizations(dataset, output_dir="visualy") 