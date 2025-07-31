import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = self.config['model']['dim']
        self.depth = self.config['model']['depth']

        # Encoder
        self.encoders = nn.ModuleList()
        in_channels = 2 # Stereo
        for i in range(self.depth):
            out_channels = dim * (2**i)
            self.encoders.append(
                nn.Sequential(
                    ConvBlock(in_channels, out_channels),
                    ConvBlock(out_channels, out_channels),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) # Downsampling
                )
            )
            in_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(in_channels, dim * (2**self.depth)),
            ConvBlock(dim * (2**self.depth), in_channels)
        )

        # TODO: Add Transformer blocks here as specified in the config.
        # This would involve creating a Transformer layer that operates on the time and frequency dimensions.

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(self.depth - 1, -1, -1):
            out_channels = dim * (2**i)
            self.decoders.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    ConvBlock(out_channels*2, out_channels),
                    ConvBlock(out_channels, out_channels)
                )
            )
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[2:]
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder[0](x) # Upsampling
            skip_connection = skip_connections[i]
            # Adjust size if necessary
            if x.shape != skip_connection.shape:
                 x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = decoder[1:](concat_skip)

        x = nn.functional.interpolate(x, size=input_shape)
        x = self.final_conv(x)
        return torch.tanh(x) # Tanh to output a mask between -1 and 1


class AudioEnhancer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # STFT and iSTFT layers
        self.stft = T.Spectrogram(
            n_fft=config['model']['stft_n_fft'],
            hop_length=config['model']['stft_hop_length'],
            win_length=config['model']['stft_win_length'],
            power=None, # for complex-valued spectrogram
        )
        self.istft = T.InverseSpectrogram(
            n_fft=config['model']['stft_n_fft'],
            hop_length=config['model']['stft_hop_length'],
            win_length=config['model']['stft_win_length'],
        )

        self.unet = Unet(config)

    def forward(self, lq_audio):
        # Get the complex-valued spectrogram
        lq_spec = self.stft(lq_audio)

        # Magnitude and phase
        lq_mag = lq_spec.abs()
        lq_phase = lq_spec.angle()

        # The U-Net expects input of shape (batch, channels, freq, time).
        # lq_mag already has this shape.

        # Generate mask
        mask = self.unet(lq_mag)

        # Apply mask to magnitude
        hq_mag = lq_mag * mask

        # Combine with original phase
        hq_spec = torch.polar(hq_mag, lq_phase)

        # Inverse STFT to get waveform
        hq_audio = self.istft(hq_spec, length=lq_audio.shape[-1])

        return hq_audio
