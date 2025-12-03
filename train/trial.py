import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from helpers import *
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import librosa

# Augmentations
from audio_transforms import(
    Compose,
    RandomTimeStretch, 
    RandomPitchShift, 
    RandomAddNoise, 
    RandomGain, 
    RandomTimeShift
)
transform_series = Compose([
    RandomTimeStretch(p=0.5),
    RandomPitchShift(p=0.5),
    RandomAddNoise(p=0.5),
    RandomGain(p=0.5),
    RandomTimeShift(p=0.5)
])
    

def collate_fn(batch):
    xs = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    xs_padded = pad_sequence(xs, batch_first=True) 
    return xs_padded, ys

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, sr=48000, transform=None):
        self.files = []
        self.labels = labels
        self.sr = sr
        self.transform = transform
        for f in file_paths:
            y, _ = librosa.load(f)
            self.files.append(y)
            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav = self.files[idx]
        if self.transform is not None:
            wav = self.transform(self.files[idx], self.sr)

        features = librosa.feature.melspectrogram(y=wav, sr=self.sr)
        return features.transpose(), self.labels[idx]
    
class AudioRNN(nn.Module):
    def __init__(self,
                 feature_dim=1,
                 proj_dim=64,
                 hidden_dim=128,
                 num_layers=2,
                 num_classes=8):
        super().__init__()

        # ---- 1. Projection 1024 → 256 ----
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU()
        )

        # ---- 2. BiLSTM ----
        self.rnn = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # ---- 3. Final classifier ----
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):

        x = self.proj(x)         
        out, _ = self.rnn(x)     
        pooled = out.mean(dim=1) 

        return self.fc(pooled)

def test(model, test_loader):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            loss = criterion(model(x), y)
            running_loss += loss.item()
            
    print(f"Test Accuracy: {correct/total:.4f}")
    return correct/total, running_loss/len(test_loader)



def train(train_loader, test_loader, num_classes=8, device="cpu"):
    print("Training")

    model = AudioRNN(feature_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    epochs = 100
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total+=y.size(0)

        epoch_loss = running_loss / len(train_loader)
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {correct/total:.4}")

        test_a, test_l = test(model, test_loader)
        train_acc.append(correct/total)
        test_acc.append(test_a)
        train_loss.append(epoch_loss)
        test_loss.append(test_l)
    graph(train_acc=train_acc, test_acc=test_acc, train_loss=train_loss, test_loss=test_loss, epochs=epochs)

    return model

files = get_file_matrix()
file_paths = matrix_to_filename(files, RAW_DATA_PATH)
labels = get_labels(files)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)
train_ds = AudioDataset(X_train, y_train, transform=transform_series)
test_ds  = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

print("finished train/test split")
model = train(train_loader,test_loader , labels)
torch.save(model, "model.pt")
