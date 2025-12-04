import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import pickle
import os
import time

# --- 1. DEFINITIONS & CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/wikiart"
PROCESSED_DIR = "processed_data"
ANNOTATIONS = pd.read_csv("data/annotations/artemis_dataset_release_v0.csv")
IMG_TRANSFORM = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

# Load Metadata
with open(f'{PROCESSED_DIR}/vocab.pkl', 'rb') as f:
    VOCAB = pickle.load(f)
with open(f'{PROCESSED_DIR}/splits.pkl', 'rb') as f:
    SPLITS = pickle.load(f)

# --- 2. DATASET CLASS ---
class ArtDataset(Dataset):
    def __init__(self, indices, transform=None):
        self.indices = indices
        self.full_ds = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        # Create a mapping from index to caption (simplified)
        # In real scenario, map image filename to caption row in CSV
        self.captions = ANNOTATIONS['utterance'].tolist() 
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Retrieve image using the subset index
        real_idx = self.indices[idx]
        img, _ = self.full_ds[real_idx]
        
        # Get Caption (Naive implementation: assumes CSV aligns with ImageFolder sorted order)
        # WARNING: Ensure CSV matches ImageFolder logic. 
        # For assignment safety, we tokenize on the fly here:
        caption_text = str(self.captions[real_idx]).lower()
        tokens = [VOCAB.get(t, VOCAB['<unk>']) for t in re.sub(r'[^\w\s]','',caption_text).split()]
        tokens = [VOCAB['<start>']] + tokens[:18] + [VOCAB['<end>']]
        
        # Pad
        seq = torch.ones(20, dtype=torch.long) * VOCAB['<pad>']
        seq[:len(tokens)] = torch.tensor(tokens)
        
        return img, seq

# --- 3. MODEL ARCHITECTURE  ---

class EncoderCNN(nn.Module):
    """Custom CNN from scratch """
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # Image is 224x224 -> pool -> 112 -> pool -> 56 -> pool -> 28
        # 128 channels * 28 * 28
        self.fc = nn.Linear(128 * 28 * 28, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        x = self.pool(self.relu(self.conv1(images)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        features = self.fc(x)
        return features

class DecoderRNN(nn.Module):
    """LSTM Decoder [cite: 2579]"""
    def __init__(self, embed_size, hidden_size, vocab_size, embed_matrix=None):
        super(DecoderRNN, self).__init__()
        
        # Embedding Layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        if embed_matrix is not None:
            # Load pre-trained weights [cite: 2571]
            # Ensure dimension match (SVD/GloVe might vary)
            if embed_matrix.shape[1] != embed_size:
                # Project if dims mismatch (e.g. FastText 300 -> 256)
                self.project_emb = nn.Linear(embed_matrix.shape[1], embed_size)
                self.raw_embed_weights = torch.tensor(embed_matrix, dtype=torch.float32).to(DEVICE)
                self.use_projection = True
            else:
                self.embed.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
                self.embed.weight.requires_grad = False # Non-trainable
                self.use_projection = False
        else:
            self.use_projection = False

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # Embed captions
        if self.use_projection:
            # Manual lookup and projection for mismatched dims
            embs = torch.nn.functional.embedding(captions, self.raw_embed_weights)
            embeddings = self.project_emb(embs)
        else:
            embeddings = self.embed(captions)

        embeddings = self.dropout(embeddings)
        
        # Concatenate image features as the "start" of the sequence (standard captioning trick)
        # features shape: (batch, embed_size) -> (batch, 1, embed_size)
        features = features.unsqueeze(1)
        
        # Input to LSTM: Image + Words (excluding <end>)
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

class ImageCaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embed_matrix=None):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, embed_matrix)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# --- 4. TRAINING ENGINE ---

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, captions in loader:
        images, captions = images.to(DEVICE), captions.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images, captions)
        
        # Calculate loss (Output: Batch, Seq, Vocab) -> (Batch*Seq, Vocab)
        # Target: (Batch, Seq) -> (Batch*Seq)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in loader:
            images, captions = images.to(DEVICE), captions.to(DEVICE)
            outputs = model(images, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

# --- 5. HYPERPARAMETER TESTING LOOP  ---

def run_experiments():
    # Load processed embeddings
    emb_glove = np.load(f'{PROCESSED_DIR}/emb_glove.npy')
    emb_fasttext = np.load(f'{PROCESSED_DIR}/emb_fasttext.npy')
    emb_tfidf = np.load(f'{PROCESSED_DIR}/emb_tfidf.npy')

    # Experiment Configs
    embeddings = {
        'GloVe': (emb_glove, 200),
        'FastText': (emb_fasttext, 300), # Dim mismatch handled in model
        'TF-IDF': (emb_tfidf, 200)
    }
    
    hidden_sizes = [256] # Keep it simple for demo, add [128] if time permits
    
    best_val_loss = float('inf')
    best_model_name = ""

    # Data Loaders
    train_data = ArtDataset(SPLITS['train'], IMG_TRANSFORM)
    val_data = ArtDataset(SPLITS['val'], IMG_TRANSFORM)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    results = []

    print(f"Starting Experiments on {DEVICE}...")

    for emb_name, (emb_matrix, emb_dim) in embeddings.items():
        for hidden_dim in hidden_sizes:
            print(f"\n--- Training: {emb_name} | Hidden: {hidden_dim} ---")
            
            # Note: We generally match the CNN output size to the Embedding dim
            model = ImageCaptionModel(embed_size=256, hidden_size=hidden_dim, vocab_size=len(VOCAB), embed_matrix=emb_matrix).to(DEVICE)
            
            criterion = nn.CrossEntropyLoss(ignore_index=VOCAB['<pad>'])
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Short training loop for selection
            for epoch in range(2): # Run more epochs (e.g., 10) in reality
                loss = train_epoch(model, train_loader, criterion, optimizer)
                val_loss = validate(model, val_loader, criterion)
                print(f"Epoch {epoch+1}: Train Loss {loss:.4f}, Val Loss {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_name = f"{emb_name}_h{hidden_dim}"
                torch.save(model.state_dict(), "best_model.pth")
            
            results.append({'model': f"{emb_name}_{hidden_dim}", 'val_loss': val_loss})

    print(f"\nBest Model: {best_model_name} with Loss: {best_val_loss}")
    return results

if __name__ == "__main__":
    import re 
    import pandas as pd 
    run_experiments()