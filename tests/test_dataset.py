import unittest
import torch
import numpy as np
import os
import soundfile as sf
import shutil

from src.dataset import MusicDataset
from src.utils import load_config

class TestMusicDataset(unittest.TestCase):

    def setUp(self):
        """Set up a dummy data directory and a dummy audio file."""
        self.test_dir = "test_data"
        os.makedirs(self.test_dir, exist_ok=True)

        self.config = load_config("configs/lq2hq_config.yaml")
        self.sample_rate = self.config['audio']['sample_rate']
        self.chunk_size = self.config['audio']['chunk_size']

        # Create a dummy silent wav file
        self.dummy_audio_path = os.path.join(self.test_dir, "dummy_audio.wav")
        dummy_audio_data = np.zeros((self.chunk_size,))
        sf.write(self.dummy_audio_path, dummy_audio_data, self.sample_rate)

    def test_dataset_loading(self):
        """Test that the dataset can be loaded and returns tensors of the correct shape."""
        dataset = MusicDataset(data_path=self.test_dir, config=self.config)
        self.assertEqual(len(dataset), 1)

        lq_audio, hq_audio = dataset[0]

        self.assertIsInstance(lq_audio, torch.Tensor)
        self.assertIsInstance(hq_audio, torch.Tensor)

        # Shape should be (num_channels, chunk_size)
        self.assertEqual(lq_audio.shape, (2, self.chunk_size))
        self.assertEqual(hq_audio.shape, (2, self.chunk_size))

    def tearDown(self):
        """Remove the dummy data directory and its contents."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
