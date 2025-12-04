import numpy as np
import librosa

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, y, sr):
        for t in self.transforms:
            y = t(y, sr)
        return y

class RandomTimeStretch:
    def __init__(self, p=0.5, min_rate=0.85, max_rate=1.15):
        self.p = p
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, y, sr):
        if np.random.rand() < self.p:
            rate = np.random.uniform(self.min_rate, self.max_rate)
            y = librosa.effects.time_stretch(y, rate=rate)
        return y

class RandomPitchShift:
    def __init__(self, p=0.5, min_steps=-3, max_steps=3):
        self.p = p
        self.min_steps = min_steps
        self.max_steps = max_steps

    def __call__(self, y, sr):
        if np.random.rand() < self.p:
            n_steps = np.random.randint(self.min_steps, self.max_steps + 1)
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
        return y

class RandomAddNoise:
    def __init__(self, p=0.5, noise_level=0.005):
        self.p = p
        self.noise_level = noise_level

    def __call__(self, y, sr):
        if np.random.rand() < self.p:
            y = y + self.noise_level * np.random.randn(len(y))
        return y

class RandomGain:
    def __init__(self, p=0.5, min_gain=0.7, max_gain=1.3):
        self.p = p
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, y, sr):
        if np.random.rand() < self.p:
            gain = np.random.uniform(self.min_gain, self.max_gain)
            y = y * gain
        return y

class RandomTimeShift:
    def __init__(self, p=0.5, max_shift=1000):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, y, sr):
        if np.random.rand() < self.p:
            shift = np.random.randint(-self.max_shift, self.max_shift)
            y = np.roll(y, shift)
        return y
