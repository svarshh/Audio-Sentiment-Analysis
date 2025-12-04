import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

class CNNEmotionClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (1, 43, 400) -> (16, 43, 400)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # (16, 21, 200)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (32, 21, 200)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # (32, 10, 100)
        )

        # flatten = 32 * 10 * 100 = 32000
        self.fc = nn.Sequential(
            nn.Linear(32000, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
def extract_features_from_waveform(x, sr, n_mfcc=40, max_len=400):
    """
    Same as extract_features, but input is already loaded waveform x.
    """
    # MFCC
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=x, sr=sr)

    # RMS
    rms = librosa.feature.rms(y=x)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(x)

    # reshape helper
    def reshape(feature, max_len):
        if feature.shape[1] < max_len:
            pad_width = max_len - feature.shape[1]
            feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            feature = feature[:, :max_len]
        return feature

    mfcc = reshape(mfcc, max_len)
    centroid = reshape(centroid, max_len)
    rms = reshape(rms, max_len)
    zcr = reshape(zcr, max_len)

    features = np.vstack([mfcc, centroid, rms, zcr])
    return torch.tensor(features).float().unsqueeze(0).unsqueeze(0)
