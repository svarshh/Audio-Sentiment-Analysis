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

if(__name__ == "__main__"):
    def train(embeddings, labels, num_classes=8, device="cpu"):
        # Split train/test manually if needed
        print("Training")
        dataset = EmbeddingDataset(embeddings, labels)
        loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
        model = AudioRNN(feature_dim=1024, hidden_dim=128, num_layers=2,num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
        epochs = 200
        for epoch in range(epochs):
            print(f"epoch: {epoch}")
            model.train()
            running_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f}")
    
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)            
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Accuracy: {correct/total:.4f}")
    
        return model

    files = get_file_matrix()
    file_paths = matrix_to_filename(files, RAW_DATA_PATH)
    labels = get_labels(files)
    embeddings = []
    EMBEDDING_FILE = 'embeddings.pt'
    print("Getting embeddings")
    
    if(os.path.exists(EMBEDDING_FILE)):
        print(f"Loading embeddings from {EMBEDDING_FILE}")
        embeddings = torch.load(EMBEDDING_FILE)
    else:
        print(f"Loading embeddings using Wav2Vec")
        c = 0
        for f in file_paths:
            embeddings.append(torch.from_numpy(get_embedding(f))
            .unsqueeze(-1).squeeze(0).squeeze(-1))
            c+=1
            print(c)
        torch.save(embeddings, EMBEDDING_FILE)
        print(f"Saved embeddings to {EMBEDDING_FILE}")
    print("Done getting embeddings")
    
    
    train_emb, test_emb, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    print("finished train/test split")
    model = train(train_emb, y_train)
    
    correct, total = 0, 0
    for emb, label in zip(test_emb, y_test):
        x = emb.unsqueeze(0).to(device) 
        y = torch.tensor([label], device=device)
        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += 1
    torch.save(model, "model.pt")
    torch.save(model.state_dict(), "model_weights.pt")
    print(f"\nAccuracy: {correct/total:.4f}")