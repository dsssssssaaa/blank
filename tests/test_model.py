import unittest
import torch

from src.model import AudioEnhancer

class TestAudioEnhancer(unittest.TestCase):

    def setUp(self):
        """Create a lightweight model configuration for testing."""
        self.config = {
            'audio': {
                'sample_rate': 44100,
                'chunk_size': 8192,
                'num_channels': 2,
            },
            'model': {
                'dim': 16,
                'depth': 2,
                'stereo': True,
                'stft_n_fft': 2048,
                'stft_hop_length': 441,
                'stft_win_length': 2048,
            },
            'training': {
                'batch_size': 1,
            }
        }
        self.batch_size = self.config['training']['batch_size']
        self.chunk_size = self.config['audio']['chunk_size']
        self.num_channels = self.config['audio']['num_channels']

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        model = AudioEnhancer(self.config)
        model.eval()

        # Create a dummy input tensor
        dummy_input = torch.randn(self.batch_size, self.num_channels, self.chunk_size)

        with torch.no_grad():
            output = model(dummy_input)

        # Check that the output has the same shape as the input
        self.assertEqual(output.shape, dummy_input.shape)

if __name__ == '__main__':
    unittest.main()
