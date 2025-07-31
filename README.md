# Audio Enhancement with U-Net

This project provides a set of scripts to train a deep learning model for audio enhancement. Specifically, it's designed to take low-quality (LQ) audio recordings, such as live concert recordings, and improve their quality, making them sound closer to high-quality (HQ) studio recordings.

The core of the project is a U-Net based model that operates on audio spectrograms to perform the enhancement.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    The required Python packages are listed in `requirements.txt`. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note on PyTorch:* The default `torch` package includes CUDA support and can be very large. If you are in a resource-constrained environment or don't have a compatible GPU, you can install a CPU-only version of PyTorch, which is much smaller. You can do this before installing the other requirements:
    ```bash
    pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    ```

## Data Preparation

To train the model, you need a dataset of high-quality (HQ) audio files.

1.  Create a directory (e.g., `data/my_hq_audio`).
2.  Place all your HQ audio files (in `.wav`, `.flac`, or `.mp3` format) inside this directory.

The training script will automatically generate low-quality (LQ) versions of the audio on-the-fly by applying a series of augmentations, such as adding noise or simulating MP3 compression.

## Training

The main training script is `src/train.py`. You can run it from the command line, providing the path to the configuration file and the data directory.

```bash
python src/train.py --config configs/lq2hq_config.yaml --data_path /path/to/your/hq_audio --epochs 50
```

### Arguments:
- `--config`: Path to the YAML configuration file (e.g., `configs/lq2hq_config.yaml`).
- `--data_path`: Path to the directory containing your high-quality audio files.
- `--checkpoint_dir`: (Optional) Directory to save model checkpoints. Defaults to `checkpoints`.
- `--epochs`: (Optional) Number of epochs to train for. Defaults to 100.

Model checkpoints will be saved in the specified checkpoint directory at the end of each epoch.

## Model Architecture

The model is an `AudioEnhancer` that wraps a U-Net. It works as follows:
1.  It takes a raw audio waveform as input.
2.  It converts the waveform to a complex spectrogram using STFT.
3.  The magnitude of the spectrogram is passed through a U-Net, which predicts a multiplicative mask.
4.  The mask is applied to the input spectrogram's magnitude.
5.  The enhanced spectrogram is converted back to a waveform using iSTFT.

The architecture of the U-Net and other parameters can be controlled through the YAML configuration file in the `configs` directory.
