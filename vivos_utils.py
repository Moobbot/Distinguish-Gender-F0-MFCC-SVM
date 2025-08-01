import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from pathlib import Path
import librosa
import librosa.display
from vivos_data_loader import VivosDataset
import os


def plot_audio_waveform(waveform, sample_rate, title="Audio Waveform", save_path=None):
    """Vẽ waveform của audio"""
    plt.figure(figsize=(12, 4))
    plt.plot(waveform.squeeze().numpy())
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Waveform saved to: {save_path}")
    
    plt.show()


def plot_spectrogram(waveform, sample_rate, title="Spectrogram", save_path=None):
    """Vẽ spectrogram của audio"""
    # Chuyển đổi sang numpy array
    audio_np = waveform.squeeze().numpy()

    # Tính spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram saved to: {save_path}")
    
    plt.show()


def analyze_audio_durations(dataset, save_path=None):
    """Phân tích độ dài của các audio files"""
    durations = []

    for i in range(min(100, len(dataset))):  # Chỉ phân tích 100 sample đầu
        sample = dataset[i]
        duration = sample['audio'].shape[1] / sample['sample_rate']
        durations.append(duration)

    durations = np.array(durations)

    print(f"Audio Duration Statistics:")
    print(f"Mean: {durations.mean():.2f} seconds")
    print(f"Std: {durations.std():.2f} seconds")
    print(f"Min: {durations.min():.2f} seconds")
    print(f"Max: {durations.max():.2f} seconds")
    print(f"Median: {np.median(durations):.2f} seconds")

    # Vẽ histogram
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Audio Durations')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Audio duration histogram saved to: {save_path}")
    
    plt.show()

    return durations


def analyze_text_lengths(dataset, save_path=None):
    """Phân tích độ dài của text"""
    text_lengths = []

    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        text_length = len(sample['text'].split())
        text_lengths.append(text_length)

    text_lengths = np.array(text_lengths)

    print(f"Text Length Statistics (words):")
    print(f"Mean: {text_lengths.mean():.2f} words")
    print(f"Std: {text_lengths.std():.2f} words")
    print(f"Min: {text_lengths.min():.2f} words")
    print(f"Max: {text_lengths.max():.2f} words")
    print(f"Median: {np.median(text_lengths):.2f} words")

    # Vẽ histogram
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Lengths')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Text length histogram saved to: {save_path}")
    
    plt.show()

    return text_lengths


def get_speaker_samples(dataset, speaker_id, num_samples=5):
    """Lấy một số sample của một speaker cụ thể"""
    samples = []
    for i in range(len(dataset)):
        if dataset.data_list[i]['speaker_id'] == speaker_id:
            samples.append(dataset[i])
            if len(samples) >= num_samples:
                break

    return samples


def visualize_speaker_samples(dataset, speaker_id, num_samples=3, save_path=None):
    """Visualize các sample của một speaker"""
    samples = get_speaker_samples(dataset, speaker_id, num_samples)

    if not samples:
        print(f"No samples found for speaker {speaker_id}")
        return

    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))

    for i, sample in enumerate(samples):
        # Waveform
        waveform = sample['audio'].squeeze().numpy()
        axes[i, 0].plot(waveform)
        axes[i, 0].set_title(f"Speaker {speaker_id} - Sample {i+1}")
        axes[i, 0].set_xlabel("Sample")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True)

        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        librosa.display.specshow(D, sr=sample['sample_rate'], ax=axes[i, 1])
        axes[i, 1].set_title(f"Spectrogram - Sample {i+1}")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Speaker samples visualization saved to: {save_path}")
    
    plt.show()

    # In text
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample['text']}")


def create_audio_preprocessing_pipeline(sample_rate=16000,
                                        normalize=True,
                                        add_noise=False,
                                        noise_level=0.01):
    """Tạo pipeline preprocessing cho audio"""

    def preprocess_audio(waveform, sr):
        # Resample nếu cần
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Normalize
        if normalize:
            waveform = waveform / torch.max(torch.abs(waveform))

        # Thêm noise nếu cần
        if add_noise:
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        return waveform

    return preprocess_audio


def extract_mel_spectrogram(waveform, sample_rate, n_mels=80, n_fft=1024, hop_length=256):
    """Trích xuất mel spectrogram"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_spec = mel_transform(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    return mel_spec_db


def create_mel_dataset(dataset, n_mels=80):
    """Tạo dataset với mel spectrogram features"""
    mel_dataset = []

    for i in range(len(dataset)):
        sample = dataset[i]
        mel_spec = extract_mel_spectrogram(
            sample['audio'],
            sample['sample_rate'],
            n_mels=n_mels
        )

        mel_sample = {
            'mel_spec': mel_spec,
            'text': sample['text'],
            'audio_id': sample['audio_id'],
            'speaker_id': sample['speaker_id'],
            'gender': sample['gender']
        }

        mel_dataset.append(mel_sample)

    return mel_dataset


def save_dataset_info(dataset, output_file="dataset_info.txt"):
    """Lưu thông tin dataset ra file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== VIVOS DATASET INFORMATION ===\n\n")

        # Thống kê cơ bản
        f.write(f"Total samples: {len(dataset)}\n")

        # Thống kê speaker
        speaker_counts, speaker_genders = dataset.get_speaker_stats()
        f.write(f"Total speakers: {len(speaker_counts)}\n\n")

        f.write("Speaker statistics:\n")
        for speaker_id, count in speaker_counts.items():
            gender = speaker_genders[speaker_id]
            f.write(f"  {speaker_id}: {count} samples ({gender})\n")

        # Thống kê gender
        gender_counts = dataset.get_gender_stats()
        f.write(f"\nGender distribution:\n")
        for gender, count in gender_counts.items():
            f.write(f"  {gender}: {count} samples\n")

        # Một số sample mẫu
        f.write(f"\nSample examples:\n")
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            f.write(
                f"  {i+1}. {sample['audio_id']}: {sample['text'][:100]}...\n")

    print(f"Dataset information saved to {output_file}")


# Demo functions
def demo_vivos_analysis():
    """Demo phân tích dataset VIVOS"""
    print("=== VIVOS DATASET ANALYSIS DEMO ===\n")

    # Load dataset
    dataset = VivosDataset("vivos", split='train')

    # Phân tích audio durations
    print("1. Analyzing audio durations...")
    durations = analyze_audio_durations(dataset)

    # Phân tích text lengths
    print("\n2. Analyzing text lengths...")
    text_lengths = analyze_text_lengths(dataset)

    # Visualize một số speaker
    print("\n3. Visualizing speaker samples...")
    speaker_ids = list(dataset.get_speaker_stats()[0].keys())[:3]
    for speaker_id in speaker_ids:
        print(f"\nVisualizing samples for speaker {speaker_id}...")
        visualize_speaker_samples(dataset, speaker_id, num_samples=2)

    # Lưu thông tin dataset
    print("\n4. Saving dataset information...")
    save_dataset_info(dataset)

    print("\n=== DEMO COMPLETED ===")


def create_all_visualizations(dataset, output_dir="visualize"):
    """Tạo tất cả các visualization và lưu vào folder"""
    # Tạo folder nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating visualizations in folder: {output_dir}")
    
    # 1. Audio duration histogram
    print("1. Creating audio duration histogram...")
    analyze_audio_durations(dataset, save_path=os.path.join(output_dir, "audio_duration_histogram.png"))
    
    # 2. Text length histogram
    print("2. Creating text length histogram...")
    analyze_text_lengths(dataset, save_path=os.path.join(output_dir, "text_length_histogram.png"))
    
    # 3. Sample audio waveforms và spectrograms
    print("3. Creating sample audio visualizations...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        
        # Waveform
        plot_audio_waveform(
            sample['audio'], 
            sample['sample_rate'],
            title=f"Audio Waveform: {sample['audio_id']}",
            save_path=os.path.join(output_dir, f"waveform_{sample['audio_id']}.png")
        )
        
        # Spectrogram
        plot_spectrogram(
            sample['audio'], 
            sample['sample_rate'],
            title=f"Spectrogram: {sample['audio_id']}",
            save_path=os.path.join(output_dir, f"spectrogram_{sample['audio_id']}.png")
        )
    
    # 4. Speaker samples visualization
    print("4. Creating speaker samples visualizations...")
    speaker_ids = list(dataset.get_speaker_stats()[0].keys())[:5]  # 5 speakers đầu tiên
    for speaker_id in speaker_ids:
        visualize_speaker_samples(
            dataset, 
            speaker_id, 
            num_samples=2,
            save_path=os.path.join(output_dir, f"speaker_{speaker_id}_samples.png")
        )
    
    # 5. Gender distribution pie chart
    print("5. Creating gender distribution chart...")
    gender_counts = dataset.get_gender_stats()
    
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%')
    plt.title('Gender Distribution in VIVOS Dataset')
    plt.savefig(os.path.join(output_dir, "gender_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Gender distribution chart saved to: {os.path.join(output_dir, 'gender_distribution.png')}")
    
    # 6. Speaker count bar chart
    print("6. Creating speaker count chart...")
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
    plt.show()
    print(f"Top speakers chart saved to: {os.path.join(output_dir, 'top_speakers_count.png')}")
    
    print(f"\nAll visualizations saved to folder: {output_dir}")
    print("Files created:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")


if __name__ == "__main__":
    # Demo cũ
    # demo_vivos_analysis()
    
    # Tạo tất cả visualizations và lưu vào folder visualize
    print("=== CREATING ALL VISUALIZATIONS ===")
    
    # Load dataset
    dataset = VivosDataset("vivos", split='train')
    
    # Tạo tất cả visualizations
    create_all_visualizations(dataset, output_dir="visualize")
