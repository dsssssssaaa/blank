import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np
import os

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, config):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.hq_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(('.wav', '.flac', '.mp3'))]
        self.chunk_size = config['audio']['chunk_size']

    def __len__(self):
        return len(self.hq_files)

    def __getitem__(self, idx):
        hq_path = self.hq_files[idx]

        try:
            hq_audio, sample_rate = torchaudio.load(hq_path)
        except Exception as e:
            print(f"Error loading file {hq_path}: {e}")
            # Return a dummy tensor to avoid crashing the training loop
            return torch.zeros(2, self.chunk_size), torch.zeros(2, self.chunk_size)

        # Resample if necessary
        if sample_rate != self.config['audio']['sample_rate']:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.config['audio']['sample_rate'])
            hq_audio = resampler(hq_audio)

        # Convert to stereo if mono
        if hq_audio.shape[0] == 1:
            hq_audio = hq_audio.repeat(2, 1)

        # Trim or pad to chunk_size
        if hq_audio.shape[1] < self.chunk_size:
            hq_audio = torch.nn.functional.pad(hq_audio, (0, self.chunk_size - hq_audio.shape[1]))
        elif hq_audio.shape[1] > self.chunk_size:
            start = random.randint(0, hq_audio.shape[1] - self.chunk_size)
            hq_audio = hq_audio[:, start:start + self.chunk_size]

        # Create LQ audio by applying augmentations
        lq_audio = self.apply_augmentations(hq_audio.clone())

        return lq_audio, hq_audio

    def apply_augmentations(self, audio):
        # This is a simplified augmentation pipeline.
        # In a real scenario, you'd implement more of the augmentations from the config.

        # With a certain probability, add gaussian noise
        if self.config['augmentations']['enable'] and self.config['augmentations']['gaussian_noise'] > random.random():
            noise = torch.randn_like(audio) * random.uniform(
                self.config['augmentations']['gaussian_noise_min_amplitude'],
                self.config['augmentations']['gaussian_noise_max_amplitude']
            )
            audio += noise

        # Other augmentations like reverb, compression, etc. would be added here.
        # For example, to simulate MP3 compression:
        if self.config['augmentations']['enable'] and self.config['augmentations']['mp3_compression'] > random.random():
             # This requires lameenc to be installed
            try:
                bitrate = random.randint(
                    self.config['augmentations']['mp3_compression_min_bitrate'],
                    self.config['augmentations']['mp3_compression_max_bitrate']
                )
                encoder = T.MP3Encoder(bitrate=bitrate)
                # This is a simplified simulation. A more accurate one would save to a buffer and reload.
                # For now, we just apply a low-pass filter as a proxy.
                audio = T.LowpassFilter(cutoff_freq=bitrate * 100)(audio)
            except Exception as e:
                # This can fail if the backend is not available.
                # For now we just print a warning
                print(f"Warning: Could not apply MP3 compression simulation: {e}")
                pass

        return audio
