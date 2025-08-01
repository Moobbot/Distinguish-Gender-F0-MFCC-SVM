import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path


class VivosDataset(Dataset):
    """
    Dataset class cho VIVOS Vietnamese Speech Corpus
    """

    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 transform=None,
                 max_length: Optional[int] = None):
        """
        Args:
            root_dir: Đường dẫn đến thư mục gốc của VIVOS dataset
            split: 'train' hoặc 'test'
            transform: Transform áp dụng cho audio
            max_length: Độ dài tối đa của audio (samples)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.max_length = max_length

        # Đường dẫn đến các thư mục
        self.waves_dir = self.root_dir / split / 'waves'
        self.prompts_file = self.root_dir / split / 'prompts.txt'
        self.genders_file = self.root_dir / split / 'genders.txt'

        # Load dữ liệu
        self._load_transcripts()
        self._load_speaker_info()
        self._create_data_list()

    def _load_transcripts(self):
        """Load transcripts từ file prompts.txt"""
        self.transcripts = {}
        with open(self.prompts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        audio_id, text = parts
                        self.transcripts[audio_id] = text

    def _load_speaker_info(self):
        """Load thông tin speaker từ file genders.txt"""
        self.speaker_info = {}
        with open(self.genders_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ')
                    if len(parts) == 2:
                        speaker_id, gender = parts
                        self.speaker_info[speaker_id] = gender

    def _create_data_list(self):
        """Tạo danh sách các sample có sẵn"""
        self.data_list = []

        # Duyệt qua tất cả các thư mục speaker
        for speaker_dir in self.waves_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name

                # Duyệt qua tất cả các file audio
                for audio_file in speaker_dir.glob('*.wav'):
                    audio_id = audio_file.stem

                    # Kiểm tra xem có transcript tương ứng không
                    if audio_id in self.transcripts:
                        self.data_list.append({
                            'audio_path': str(audio_file),
                            'audio_id': audio_id,
                            'speaker_id': speaker_id,
                            'text': self.transcripts[audio_id],
                            'gender': self.speaker_info.get(speaker_id, 'unknown')
                        })

        print(f"Loaded {len(self.data_list)} samples from {self.split} set")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Lấy một sample"""
        sample = self.data_list[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(sample['audio_path'])

        # Áp dụng transform nếu có
        if self.transform:
            waveform = self.transform(waveform)

        # Cắt hoặc pad audio nếu cần
        if self.max_length:
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                # Pad với zeros
                padding = torch.zeros(1, self.max_length - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)

        return {
            'audio': waveform,
            'sample_rate': sample_rate,
            'text': sample['text'],
            'audio_id': sample['audio_id'],
            'speaker_id': sample['speaker_id'],
            'gender': sample['gender']
        }

    def get_speaker_stats(self):
        """Thống kê theo speaker"""
        speaker_counts = {}
        speaker_genders = {}

        for sample in self.data_list:
            speaker_id = sample['speaker_id']
            if speaker_id not in speaker_counts:
                speaker_counts[speaker_id] = 0
                speaker_genders[speaker_id] = sample['gender']
            speaker_counts[speaker_id] += 1

        return speaker_counts, speaker_genders

    def get_gender_stats(self):
        """Thống kê theo giới tính"""
        gender_counts = {}
        for sample in self.data_list:
            gender = sample['gender']
            if gender not in gender_counts:
                gender_counts[gender] = 0
            gender_counts[gender] += 1

        return gender_counts


def create_vivos_dataloader(root_dir: str,
                            split: str = 'train',
                            batch_size: int = 32,
                            shuffle: bool = True,
                            num_workers: int = 4,
                            max_length: Optional[int] = None,
                            transform=None):
    """
    Tạo DataLoader cho VIVOS dataset

    Args:
        root_dir: Đường dẫn đến thư mục gốc của VIVOS dataset
        split: 'train' hoặc 'test'
        batch_size: Kích thước batch
        shuffle: Có shuffle data không
        num_workers: Số worker cho DataLoader
        max_length: Độ dài tối đa của audio (samples)
        transform: Transform áp dụng cho audio

    Returns:
        DataLoader object
    """
    dataset = VivosDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return dataloader


def collate_fn(batch):
    """
    Collate function cho DataLoader
    """
    # Tách các thành phần
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    audio_ids = [item['audio_id'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    genders = [item['gender'] for item in batch]
    sample_rates = [item['sample_rate'] for item in batch]

    # Stack audio tensors
    audios = torch.stack(audios)

    return {
        'audio': audios,
        'text': texts,
        'audio_id': audio_ids,
        'speaker_id': speaker_ids,
        'gender': genders,
        'sample_rate': sample_rates[0]  # Giả sử tất cả có cùng sample rate
    }


# Ví dụ sử dụng
if __name__ == "__main__":
    # Đường dẫn đến dataset
    vivos_root = "vivos"

    # Tạo dataset
    train_dataset = VivosDataset(vivos_root, split='train')
    test_dataset = VivosDataset(vivos_root, split='test')

    # In thống kê
    print("=== TRAIN SET STATS ===")
    speaker_counts, speaker_genders = train_dataset.get_speaker_stats()
    print(f"Total speakers: {len(speaker_counts)}")
    print(f"Total samples: {len(train_dataset)}")

    gender_counts = train_dataset.get_gender_stats()
    print(f"Gender distribution: {gender_counts}")

    print("\n=== TEST SET STATS ===")
    speaker_counts_test, speaker_genders_test = test_dataset.get_speaker_stats()
    print(f"Total speakers: {len(speaker_counts_test)}")
    print(f"Total samples: {len(test_dataset)}")

    gender_counts_test = test_dataset.get_gender_stats()
    print(f"Gender distribution: {gender_counts_test}")

    # Test lấy một sample
    sample = train_dataset[0]
    print(f"\nSample audio shape: {sample['audio'].shape}")
    print(f"Sample text: {sample['text']}")
    print(f"Speaker: {sample['speaker_id']} ({sample['gender']})")

    # Tạo DataLoader
    train_loader = create_vivos_dataloader(
        root_dir=vivos_root,
        split='train',
        batch_size=4,
        shuffle=True
    )

    # Test DataLoader
    for batch in train_loader:
        print(f"\nBatch audio shape: {batch['audio'].shape}")
        print(f"Batch texts: {batch['text']}")
        break
