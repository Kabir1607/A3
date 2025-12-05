import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle
import re

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM_Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda': print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else: print("⚠️ CPU Mode")

BATCH_SIZE = 64
EPOCHS = 8
EMBED_DIM = 100 # Matches GloVe 100d
HIDDEN_DIM = 256

# --- DATASET ---
class ArtDataset(Dataset):
    def __init__(self, pkl_path, transform):
        self.df = pd.read_pickle(pkl_path)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: image = Image.open(row['abs_path']).convert("RGB")
        except: image = Image.new('RGB', (224, 224))
        image = self.transform(image)
        caption = torch.tensor(row['sequence'], dtype=torch.long)
        return image, caption

# --- MODEL ---
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_mat, freeze_emb=True):
        super().__init__()
        # Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Decoder
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embed_mat, dtype=torch.float32), freeze=freeze_emb)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)
        self.img_proj = nn.Linear(HIDDEN_DIM, EMBED_DIM)

    def forward(self, img, cap):
        img_feat = self.img_proj(self.cnn(img)).unsqueeze(1)
        cap_emb = self.embed(cap[:, :-1]) 
        inputs = torch.cat((img_feat, cap_emb), dim=1)
        out, _ = self.lstm(inputs)
        return self.fc(out)

# --- TRAIN GLOVE ---
def train():
    print("Loading Data...")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_loader = DataLoader(ArtDataset(os.path.join(PROCESSED_DIR, "train.pkl"), transform), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(ArtDataset(os.path.join(PROCESSED_DIR, "val.pkl"), transform), batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)
    
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f: vocab = pickle.load(f)

    # LOAD GLOVE MATRIX
    try:
        glove_mat = np.load(os.path.join(PROCESSED_DIR, "emb_glove.npy"))
    except:
        print("GloVe matrix not found! Using random.")
        glove_mat = np.zeros((len(vocab), EMBED_DIM))
    
    print(f"\n>>> Training GloVe Model...")
    # freeze_emb=True keeps GloVe weights static (as per assignment requirements)
    model = CaptionModel(len(vocab), glove_mat, freeze_emb=True).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_loss = float('inf')
    
    for e in range(EPOCHS):
        model.train()
        train_loss = 0
        for img, cap in train_loader:
            img, cap = img.to(DEVICE), cap.to(DEVICE)
            optim.zero_grad()
            pred = model(img, cap)
            loss = crit(pred.reshape(-1, len(vocab)), cap.reshape(-1))
            loss.backward()
            optim.step()
            train_loss += loss.item()
        
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, cap in val_loader:
                img, cap = img.to(DEVICE), cap.to(DEVICE)
                pred = model(img, cap)
                val_loss += crit(pred.reshape(-1, len(vocab)), cap.reshape(-1)).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"  Epoch {e+1}: Val Loss {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model_GloVe.pth"))

    print(f"\n✅ GloVe Training Done. Saved to {RESULTS_DIR}/best_model_GloVe.pth")

if __name__ == "__main__":
    train()