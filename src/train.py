import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

from src.dataset import MusicDataset
from src.model import AudioEnhancer
from src.utils import load_config

def train(config, data_path, checkpoint_dir, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = MusicDataset(data_path=data_path, config=config)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)

    # Create model
    model = AudioEnhancer(config).to(device)

    # Create optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])

    # Using a simple L1 loss on the waveform as a starting point.
    # For better results, the multi-STFT loss from the config should be implemented.
    criterion = nn.L1Loss()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for lq_audio, hq_audio in progress_bar:
            lq_audio, hq_audio = lq_audio.to(device), hq_audio.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predicted_hq_audio = model(lq_audio)

            # Calculate loss
            loss = criterion(predicted_hq_audio, hq_audio)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an audio enhancement model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory with high-quality audio files.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")

    args = parser.parse_args()

    config = load_config(args.config)

    train(config, args.data_path, args.checkpoint_dir, args.epochs)
