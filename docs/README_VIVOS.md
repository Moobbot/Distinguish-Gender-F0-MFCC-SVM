# VIVOS Dataset Loader

Code để đọc và xử lý dataset VIVOS (Vietnamese - Voices of Southern Corpus for Speech Recognition).

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc Dataset

VIVOS dataset có cấu trúc như sau:
```
vivos/
├── train/
│   ├── waves/
│   │   ├── VIVOSSPK01/
│   │   │   ├── VIVOSSPK01_R001.wav
│   │   │   ├── VIVOSSPK01_R002.wav
│   │   │   └── ...
│   │   ├── VIVOSSPK02/
│   │   └── ...
│   ├── prompts.txt
│   └── genders.txt
└── test/
    ├── waves/
    ├── prompts.txt
    └── genders.txt
```

## Sử dụng cơ bản

### 1. Load dataset

```python
from vivos_data_loader import VivosDataset

# Load train set
train_dataset = VivosDataset("vivos", split='train')

# Load test set
test_dataset = VivosDataset("vivos", split='test')

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

### 2. Lấy một sample

```python
# Lấy sample đầu tiên
sample = train_dataset[0]

print(f"Audio shape: {sample['audio'].shape}")
print(f"Sample rate: {sample['sample_rate']}")
print(f"Text: {sample['text']}")
print(f"Speaker: {sample['speaker_id']}")
print(f"Gender: {sample['gender']}")
```

### 3. Tạo DataLoader

```python
from vivos_data_loader import create_vivos_dataloader

# Tạo DataLoader cho training
train_loader = create_vivos_dataloader(
    root_dir="vivos",
    split='train',
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Sử dụng DataLoader
for batch in train_loader:
    audio = batch['audio']  # Shape: (batch_size, 1, audio_length)
    texts = batch['text']   # List of texts
    speaker_ids = batch['speaker_id']
    break
```

## Phân tích dữ liệu

### 1. Thống kê cơ bản

```python
from vivos_utils import analyze_audio_durations, analyze_text_lengths

# Phân tích độ dài audio
durations = analyze_audio_durations(train_dataset)

# Phân tích độ dài text
text_lengths = analyze_text_lengths(train_dataset)
```

### 2. Thống kê theo speaker

```python
# Lấy thống kê speaker
speaker_counts, speaker_genders = train_dataset.get_speaker_stats()

print("Speaker statistics:")
for speaker_id, count in speaker_counts.items():
    gender = speaker_genders[speaker_id]
    print(f"  {speaker_id}: {count} samples ({gender})")
```

### 3. Thống kê theo giới tính

```python
# Lấy thống kê gender
gender_counts = train_dataset.get_gender_stats()

print("Gender distribution:")
for gender, count in gender_counts.items():
    print(f"  {gender}: {count} samples")
```

## Visualization

### 1. Vẽ waveform

```python
from vivos_utils import plot_audio_waveform

sample = train_dataset[0]
plot_audio_waveform(
    sample['audio'], 
    sample['sample_rate'],
    title=f"Audio: {sample['audio_id']}"
)
```

### 2. Vẽ spectrogram

```python
from vivos_utils import plot_spectrogram

sample = train_dataset[0]
plot_spectrogram(
    sample['audio'], 
    sample['sample_rate'],
    title=f"Spectrogram: {sample['audio_id']}"
)
```

### 3. Visualize samples của một speaker

```python
from vivos_utils import visualize_speaker_samples

# Visualize 3 samples của speaker VIVOSSPK01
visualize_speaker_samples(train_dataset, "VIVOSSPK01", num_samples=3)
```

## Audio Preprocessing

### 1. Tạo preprocessing pipeline

```python
from vivos_utils import create_audio_preprocessing_pipeline

# Tạo pipeline với resample về 16kHz và normalize
preprocess_fn = create_audio_preprocessing_pipeline(
    sample_rate=16000,
    normalize=True,
    add_noise=False
)

# Áp dụng cho dataset
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    processed_audio = preprocess_fn(sample['audio'], sample['sample_rate'])
    # Sử dụng processed_audio...
```

### 2. Trích xuất mel spectrogram

```python
from vivos_utils import extract_mel_spectrogram

sample = train_dataset[0]
mel_spec = extract_mel_spectrogram(
    sample['audio'],
    sample['sample_rate'],
    n_mels=80
)

print(f"Mel spectrogram shape: {mel_spec.shape}")
```

## Demo hoàn chỉnh

Chạy demo để xem tất cả các tính năng:

```bash
python vivos_data_loader.py
python vivos_utils.py
```

## Tạo và lưu visualizations

### Tạo tất cả visualizations và lưu vào folder

```python
from vivos_utils import create_all_visualizations, VivosDataset

# Load dataset
dataset = VivosDataset("vivos", split='train')

# Tạo tất cả visualizations và lưu vào folder visualy
create_all_visualizations(dataset, output_dir="visualy")
```

### Lưu từng visualization riêng lẻ

```python
from vivos_utils import plot_audio_waveform, plot_spectrogram, analyze_audio_durations

# Lưu waveform
sample = dataset[0]
plot_audio_waveform(
    sample['audio'], 
    sample['sample_rate'],
    title="Audio Waveform",
    save_path="visualy/waveform.png"
)

# Lưu spectrogram
plot_spectrogram(
    sample['audio'], 
    sample['sample_rate'],
    title="Spectrogram",
    save_path="visualy/spectrogram.png"
)

# Lưu histogram độ dài audio
analyze_audio_durations(dataset, save_path="visualy/audio_duration_histogram.png")
```

## Thông tin Dataset

- **Tổng thời lượng**: 15 giờ ghi âm
- **Ngôn ngữ**: Tiếng Việt (Nam Bộ)
- **Số speaker**: 46 speakers (train) + 19 speakers (test)
- **Định dạng audio**: WAV
- **Sample rate**: 16kHz
- **License**: CC BY-NC-SA 4.0

## Tính năng chính

1. **Load dữ liệu hiệu quả**: Sử dụng PyTorch Dataset/DataLoader
2. **Phân tích thống kê**: Độ dài audio, text, phân bố speaker/gender
3. **Visualization**: Waveform, spectrogram, phân tích speaker
4. **Lưu ảnh**: Tự động lưu các visualization vào folder
5. **Preprocessing**: Resample, normalize, mel spectrogram
6. **Flexible**: Dễ dàng mở rộng và tùy chỉnh

## Các file visualization được tạo

Khi chạy `create_all_visualizations()`, các file sau sẽ được tạo trong folder `visualy/`:

- `audio_duration_histogram.png`: Histogram độ dài audio
- `text_length_histogram.png`: Histogram độ dài text
- `waveform_*.png`: Waveform của các sample audio
- `spectrogram_*.png`: Spectrogram của các sample audio
- `speaker_*_samples.png`: Visualization samples của từng speaker
- `gender_distribution.png`: Biểu đồ phân bố giới tính
- `top_speakers_count.png`: Biểu đồ số lượng samples của top speakers

## Lưu ý

- Đảm bảo dataset VIVOS đã được giải nén đúng cấu trúc
- Cần đủ RAM để load toàn bộ dataset (có thể sử dụng lazy loading)
- Có thể cần điều chỉnh `num_workers` tùy theo CPU
- Với dataset lớn, nên sử dụng `max_length` để giới hạn độ dài audio 