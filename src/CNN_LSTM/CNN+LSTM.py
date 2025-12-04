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

# ==========================================
# 1. CONFIGURATION & GPU CHECK
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM_Results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- GPU DIAGNOSTIC ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"\n✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    PIN_MEMORY = True # Faster CPU->GPU transfer
else:
    DEVICE = torch.device("cpu")
    PIN_MEMORY = False
    print("\n⚠️  GPU NOT DETECTED! Training will be slow.")
    print("   Run this command to fix it:")
    print("   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n")

BATCH_SIZE = 32
EPOCHS = 5
EMBED_DIM = 256
HIDDEN_DIM = 256

# --- PATH FINDING ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
possible_data_paths = [
    os.path.join(PROJECT_ROOT, "Initial_Artworks_folder"),
    os.path.join(PROJECT_ROOT, "Data", "wikiart"),
    os.path.join(SCRIPT_DIR, "wikiart")
]
DATA_ROOT = next((p for p in possible_data_paths if os.path.exists(p)), None)

if DATA_ROOT:
    print(f"Training Data Source: {DATA_ROOT}")
else:
    print("WARNING: Image folder not found. Check paths.")

# ==========================================
# 2. DATASET CLASS
# ==========================================
class ArtDataset(Dataset):
    def __init__(self, dataframe_path, transform=None):
        self.df = pd.read_pickle(dataframe_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['abs_image_path']).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224)) # Fallback
            
        if self.transform:
            image = self.transform(image)
        
        caption = torch.tensor(row['sequence'], dtype=torch.long)
        return image, caption

# ==========================================
# 3. MODEL ARCHITECTURE
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
        features = self.conv_layers(images)
        features = self.flatten(features)
        features = self.fc(features)
        return self.relu(features)

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, embed_matrix=None):
        super(DecoderLSTM, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        if embed_matrix is not None:
            matrix_dim = embed_matrix.shape[1]
            if matrix_dim != embed_size:
                # If dimensions differ (e.g. 200 vs 256), learn a projection
                self.project_emb = nn.Linear(matrix_dim, embed_size)
                # IMPORTANT: register_buffer ensures this moves to GPU automatically
                self.register_buffer("raw_embed_weights", torch.tensor(embed_matrix, dtype=torch.float32))
                self.use_projection = True
            else:
                self.embed = nn.Embedding.from_pretrained(torch.tensor(embed_matrix, dtype=torch.float32), freeze=True)
                self.use_projection = False
        else:
            self.use_projection = False
            
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1] # Remove <end> token
        
        if self.use_projection:
            # Manual lookup on the buffer
            raw_emb = torch.nn.functional.embedding(captions, self.raw_embed_weights)
            embeddings = self.project_emb(raw_emb)
        else:
            embeddings = self.embed(captions)
            
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), dim=1)
        
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, embed_matrix=None):
        super(CNNtoLSTM, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers, embed_matrix)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# ==========================================
# 4. TRAINING FUNCTION
# ==========================================
def train_model(model_name, embed_matrix, train_loader, val_loader, vocab_size):
    print(f"\n--- Training Configuration: {model_name} ---")
    
    model = CNNtoLSTM(EMBED_DIM, HIDDEN_DIM, vocab_size, num_layers=1, embed_matrix=embed_matrix).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, caps in train_loader:
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(imgs, caps)
            loss = criterion(outputs.reshape(-1, vocab_size), caps.reshape(-1))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, caps in val_loader:
                imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)
                outputs = model(imgs, caps)
                val_loss += criterion(outputs.reshape(-1, vocab_size), caps.reshape(-1)).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(RESULTS_DIR, f"model_{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)

    return best_val_loss

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Starting CNN + LSTM Hyperparameter Testing ---")
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # Use pin_memory=True for GPU speedup
    train_loader = DataLoader(ArtDataset(os.path.join(PROCESSED_DIR, "train_data.pkl"), transform), 
                              batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(ArtDataset(os.path.join(PROCESSED_DIR, "val_data.pkl"), transform), 
                            batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
    
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    
    embeddings_to_test = {}
    try: embeddings_to_test['TF-IDF'] = np.load(os.path.join(PROCESSED_DIR, "emb_tfidf_pca.npy"))
    except: print("TF-IDF matrix not found.")
    try: embeddings_to_test['GloVe'] = np.load(os.path.join(PROCESSED_DIR, "emb_glove.npy"))
    except: print("GloVe matrix not found.")
    try: embeddings_to_test['FastText'] = np.load(os.path.join(PROCESSED_DIR, "emb_fasttext.npy"))
    except: print("FastText matrix not found.")

    results_log = []
    best_overall_loss = float('inf')
    best_embedding_type = ""
    
    for name, matrix in embeddings_to_test.items():
        val_loss = train_model(name, matrix, train_loader, val_loader, len(vocab))
        results_log.append(f"Model: {name} | Best Val Loss: {val_loss:.4f}")
        
        if val_loss < best_overall_loss:
            best_overall_loss = val_loss
            best_embedding_type = name
            
    log_path = os.path.join(RESULTS_DIR, "tuning_results.txt")
    with open(log_path, "w") as f:
        f.write("CNN + LSTM RESULTS\n======================\n")
        for line in results_log: f.write(line + "\n")
        f.write(f"\nWINNER: {best_embedding_type}\n")

    print(f"\n>>> Winner: {best_embedding_type}. Saved to {RESULTS_DIR}")