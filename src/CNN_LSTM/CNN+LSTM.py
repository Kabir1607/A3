import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 

# Inputs
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
ANNOTATIONS_PATH = next((p for p in [
    os.path.join(SCRIPT_DIR, "artemis_dataset_release_v0.csv"),
    os.path.join(PROJECT_ROOT, "Data", "artemis_dataset_release_v0.csv")
] if os.path.exists(p)), None)

# Image Data Path
possible_data_paths = [
    os.path.join(SCRIPT_DIR, "Initial_Artworks_folder"),
    os.path.join(PROJECT_ROOT, "Initial_Artworks_folder"),
    os.path.join(PROJECT_ROOT, "Data", "wikiart"),
    os.path.join(SCRIPT_DIR, "wikiart")
]
DATA_ROOT = next((p for p in possible_data_paths if os.path.exists(p)), None)

# Output
RESULTS_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM_Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hardware Check
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"\n✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    PIN_MEMORY = True
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False
    print("\n⚠️  GPU NOT DETECTED. Training will be slow.")

BATCH_SIZE = 32
EPOCHS = 3 # Keep low for testing
EMBED_DIM = 256
HIDDEN_DIM = 256

# ==========================================
# 2. DATASET
# ==========================================
class ArtDataset(Dataset):
    def __init__(self, indices, transform=None):
        self.indices = indices
        self.transform = transform
        
        # Load Captions
        if ANNOTATIONS_PATH:
            self.df_caps = pd.read_csv(ANNOTATIONS_PATH)
            self.captions = self.df_caps['utterance'].tolist()
        else:
            self.captions = []
            
        # Image Dataset
        if DATA_ROOT:
            self.img_ds = datasets.ImageFolder(root=DATA_ROOT)
        else:
            self.img_ds = None

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Image
        if self.img_ds:
            img, _ = self.img_ds[real_idx]
        else:
            img = Image.new('RGB', (224, 224)) # Fallback
            
        if self.transform:
            img = self.transform(img)
            
        # Caption
        if self.captions:
            cap_text = str(self.captions[real_idx]).lower()
            tokens = [VOCAB.get(t, VOCAB['<unk>']) for t in re.sub(r'[^\w\s]','',cap_text).split()]
            tokens = [VOCAB['<start>']] + tokens[:18] + [VOCAB['<end>']]
            
            seq = torch.ones(20, dtype=torch.long) * VOCAB['<pad>']
            seq[:len(tokens)] = torch.tensor(tokens)
        else:
            seq = torch.zeros(20, dtype=torch.long)
            
        return img, seq

# ==========================================
# 3. MODEL
# ==========================================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 14 * 14, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        x = self.conv_layers(images)
        x = self.flatten(x)
        x = self.fc(x)
        return self.relu(x)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embed_matrix=None):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        if embed_matrix is not None:
            matrix_dim = embed_matrix.shape[1]
            if matrix_dim != embed_size:
                self.proj = nn.Linear(matrix_dim, embed_size)
                self.register_buffer("raw_emb", torch.tensor(embed_matrix, dtype=torch.float32))
                self.use_proj = True
            else:
                self.embed = nn.Embedding.from_pretrained(torch.tensor(embed_matrix, dtype=torch.float32), freeze=True)
                self.use_proj = False
        else:
            self.use_proj = False

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1]
        
        if self.use_proj:
            embs = torch.nn.functional.embedding(captions, self.raw_emb)
            embeddings = self.proj(embs)
        else:
            embeddings = self.embed(captions)
            
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)

class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embed_matrix):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, embed_matrix=embed_matrix)
    def forward(self, img, cap):
        return self.decoder(self.encoder(img), cap)

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Loading Data Resources...")
    
    # Load artifacts
    with open(os.path.join(PROCESSED_DIR, 'vocab.pkl'), 'rb') as f:
        VOCAB = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, 'splits.pkl'), 'rb') as f:
        SPLITS = pickle.load(f)

    # Load Embeddings
    embeddings = {}
    try: embeddings['TF-IDF'] = np.load(os.path.join(PROCESSED_DIR, 'emb_tfidf.npy'))
    except: print("  - TF-IDF missing.")
    try: embeddings['GloVe'] = np.load(os.path.join(PROCESSED_DIR, 'emb_glove.npy'))
    except: print("  - GloVe missing.")
    try: embeddings['FastText'] = np.load(os.path.join(PROCESSED_DIR, 'emb_fasttext.npy'))
    except: print("  - FastText missing.")

    # Dataloaders
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    if len(SPLITS['train']) > 0:
        train_loader = DataLoader(ArtDataset(SPLITS['train'], transform), batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(ArtDataset(SPLITS['val'], transform), batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
    else:
        print("CRITICAL: No training data found in splits.pkl.")
        exit()

    results_log = []
    best_loss = float('inf')
    best_name = ""

    print("\n--- Starting Hyperparameter Search ---")
    
    for name, matrix in embeddings.items():
        print(f"\n> Testing Embedding: {name}")
        model = CNNtoLSTM(EMBED_DIM, HIDDEN_DIM, len(VOCAB), matrix).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for img, cap in train_loader:
                img, cap = img.to(DEVICE), cap.to(DEVICE)
                optimizer.zero_grad()
                out = model(img, cap)
                loss = criterion(out.reshape(-1, len(VOCAB)), cap.reshape(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for img, cap in val_loader:
                    img, cap = img.to(DEVICE), cap.to(DEVICE)
                    out = model(img, cap)
                    val_loss += criterion(out.reshape(-1, len(VOCAB)), cap.reshape(-1)).item()
            
            avg_val = val_loss / len(val_loader)
            print(f"  Epoch {epoch+1}: Val Loss {avg_val:.4f}")
            
            if avg_val < best_loss:
                best_loss = avg_val
                best_name = name
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"best_model_{name}.pth"))

        results_log.append(f"Embedding: {name} | Final Val Loss: {avg_val:.4f}")

    # Save Results Log
    with open(os.path.join(RESULTS_DIR, "hyperparameter_results.txt"), "w") as f:
        f.write("CNN + LSTM RESULTS\n==================\n")
        for line in results_log: f.write(line + "\n")
        f.write(f"\nWINNER: {best_name} (Loss: {best_loss:.4f})\n")
    
    print(f"\n✅ Done! Best model saved to {RESULTS_DIR}/best_model_{best_name}.pth")