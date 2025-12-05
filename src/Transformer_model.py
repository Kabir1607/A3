import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle
import re
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM", "processed_data") 
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Transformer_Results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_transformer.pth")

# Robust Data Path Finder
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
possible_img_paths = [
    os.path.join(SCRIPT_DIR, "Initial_Artworks_folder"),
    os.path.join(SCRIPT_DIR, "CNN_LSTM", "Initial_Artworks_folder"),
    os.path.join(PROJECT_ROOT, "Initial_Artworks_folder")
]
IMG_DIR = next((p for p in possible_img_paths if os.path.exists(p)), None)

os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10          # Enough to converge with pre-trained weights
EMBED_DIM = 100      # Matches GloVe 100d
NUM_HEADS = 4        # 4 Heads for better attention
HIDDEN_DIM = 256     # Transformer FF dimension
NUM_LAYERS = 2       # Depth
DROPOUT = 0.4        # High dropout to prevent overfitting on small data
MAX_LEN = 20
VOCAB_SIZE = 5000

# ==========================================
# 2. DATASET
# ==========================================
class ArtDataset(Dataset):
    def __init__(self, pkl_path, transform):
        self.df = pd.read_pickle(pkl_path)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Image
        try: image = Image.open(row['abs_path']).convert("RGB")
        except: image = Image.new('RGB', (224, 224))
        image = self.transform(image)
        # Caption
        caption = torch.tensor(row['sequence'], dtype=torch.long)
        return image, caption

# ==========================================
# 3. TRANSFORMER MODEL
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerCaptioner(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, vocab_size, embed_matrix=None):
        super().__init__()
        
        # 1. Image Encoder (CNN to patches logic simulated via Linear projection)
        # We use a ResNet-style feature extractor or simple CNN. 
        # For "From Scratch" assignment requirement, we use a custom CNN block.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, embed_dim), # Project to embedding size
            nn.ReLU()
        )
        
        # 2. Text Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embed_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embed_matrix, dtype=torch.float32))
            # We allow fine-tuning (freeze=False) so it adapts to Art terms
        
        self.pos_encoder = PositionalEncoding(embed_dim, MAX_LEN)
        
        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=DROPOUT, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, images, captions):
        # Images: (Batch, C, H, W) -> (Batch, Embed_Dim) -> (Batch, 1, Embed_Dim)
        img_features = self.cnn(images).unsqueeze(1)
        
        # Captions: (Batch, Seq_Len) -> (Batch, Seq_Len, Embed_Dim)
        # Shifted right for teacher forcing (remove <end>)
        tgt = self.embedding(captions[:, :-1]) 
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        
        # Causal Mask (Prevent peeking future)
        sz = tgt.shape[1]
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(DEVICE)
        
        # Decoder
        # memory = image features
        output = self.transformer_decoder(tgt, img_features, tgt_mask=mask)
        return self.fc_out(output)

# ==========================================
# 4. GENERATION (BEAM SEARCH + PENALTIES)
# ==========================================
def generate_caption(model, image, vocab, idx2word, beam_width=3, alpha=0.9):
    model.eval()
    with torch.no_grad():
        # Encode Image
        img = image.unsqueeze(0).to(DEVICE)
        img_memory = model.cnn(img).unsqueeze(1)
        
        # Start with <start>
        # Tuple: (log_prob, sequence_indices)
        candidates = [(0.0, [vocab['<start>']])]
        
        for _ in range(MAX_LEN):
            next_candidates = []
            for score, seq in candidates:
                if seq[-1] == vocab['<end>']:
                    next_candidates.append((score, seq))
                    continue
                
                # Prepare input sequence
                tgt_inp = torch.tensor([seq], dtype=torch.long).to(DEVICE)
                tgt_emb = model.embedding(tgt_inp)
                tgt_emb = model.pos_encoder(tgt_emb)
                
                # Pass through Transformer
                # Note: Transformers re-process the whole sequence each step
                out = model.transformer_decoder(tgt_emb, img_memory)
                logits = model.fc_out(out[:, -1, :]) # Take last step
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # Top K
                top_k_probs, top_k_ids = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    word_idx = top_k_ids[0][i].item()
                    p = top_k_probs[0][i].item()
                    
                    # --- PENALTIES ---
                    # 1. Repetition Penalty: Hard block immediate repeats
                    if len(seq) > 0 and word_idx == seq[-1]: p -= 10.0
                    # 2. 'Bad Word' Penalty: Block <unk>
                    if word_idx == vocab['<unk>']: p -= 10.0
                    
                    next_candidates.append((score + p, seq + [word_idx]))
            
            # Sort and Prune
            candidates = sorted(next_candidates, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # Stop if all finished
            if all(c[1][-1] == vocab['<end>'] for c in candidates): break
            
    # Length Normalization (Prefer short captions? alpha < 1.0)
    # Score = log_prob / (length ^ alpha)
    best_seq = max(candidates, key=lambda x: x[0] / (len(x[1])**alpha))[1]
    
    words = [idx2word.get(i, '') for i in best_seq if i not in [vocab['<start>'], vocab['<end>']]]
    return " ".join(words)

# ==========================================
# 5. TRAINING & EVALUATION
# ==========================================
def main():
    print("--- Starting Final Transformer Training ---")
    
    # Load Data
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_loader = DataLoader(ArtDataset(os.path.join(PROCESSED_DIR, "train.pkl"), transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f: vocab = pickle.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    
    # Load Embeddings (GloVe)
    try:
        embed_matrix = np.load(os.path.join(PROCESSED_DIR, "emb_glove.npy"))
        print("Loaded GloVe Embeddings.")
    except:
        print("GloVe not found. Using Random.")
        embed_matrix = None

    # Init Model
    model = TransformerCaptioner(EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, len(vocab), embed_matrix).to(DEVICE)
    
    # Loss with Label Smoothing (Helps generalization)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'], label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for img, cap in train_loader:
            img, cap = img.to(DEVICE), cap.to(DEVICE)
            optimizer.zero_grad()
            
            output = model(img, cap) # (Batch, Seq, Vocab)
            target = cap[:, 1:]      # Shifted target
            
            loss = criterion(output.reshape(-1, len(vocab)), target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    # Save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Evaluate
    print("\n--- Evaluation (Beam Search) ---")
    model.eval()
    
    # 1. Sample Captions
    samples = test_df.sample(5)
    with open(os.path.join(RESULTS_DIR, "samples.txt"), "w") as f:
        for _, row in samples.iterrows():
            if not os.path.exists(row['abs_path']): continue
            img = Image.open(row['abs_path']).convert("RGB")
            pred = generate_caption(model, transform(img), vocab, idx2word)
            log = f"Ref: {row['utterance']}\nPred: {pred}\n{'-'*20}\n"
            print(log)
            f.write(log)
            
    # 2. BLEU Score
    scores = []
    for _, row in test_df.head(200).iterrows():
        try:
            if not os.path.exists(row['abs_path']): continue
            img = Image.open(row['abs_path']).convert("RGB")
            pred = generate_caption(model, transform(img), vocab, idx2word).split()
            ref = re.sub(r'[^\w\s]', '', str(row['utterance']).lower()).split()
            score = sentence_bleu([ref], pred, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
            scores.append(score)
        except: pass
        
    print(f"Final BLEU-1 Score: {np.mean(scores):.4f}")
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"BLEU-1: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()