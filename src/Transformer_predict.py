import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle
import math

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM", "processed_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Transformer_Results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_transformer.pth")
VOCAB_PATH = os.path.join(PROCESSED_DIR, "vocab.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 100
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_LAYERS = 2
MAX_LEN = 20

# --- MODEL ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerCaptioner(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, EMBED_DIM),
            nn.ReLU()
        )
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_encoder = PositionalEncoding(EMBED_DIM, MAX_LEN)
        decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=HIDDEN_DIM, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, images, captions):
        # Placeholder forward for state_dict loading
        return None

# --- PREDICT ---
# Added 'emotion' argument compatibility
def predict_caption(image_path, emotion=None, k=3):
    with open(VOCAB_PATH, "rb") as f: vocab = pickle.load(f)
    idx2word = {v: k for k, v in vocab.items()}

    model = TransformerCaptioner(len(vocab)).to(DEVICE)
    if not os.path.exists(MODEL_PATH): return ["Model not found"]
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    try: img = Image.open(image_path).convert("RGB")
    except: return ["Error loading image"]
    
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_memory = model.cnn(img).unsqueeze(1)
        candidates = [(0.0, [vocab['<start>']])]
        
        for _ in range(MAX_LEN):
            next_cands = []
            for score, seq in candidates:
                if seq[-1] == vocab['<end>']:
                    next_cands.append((score, seq))
                    continue
                
                tgt = torch.tensor([seq], dtype=torch.long).to(DEVICE)
                tgt_emb = model.pos_encoder(model.embedding(tgt))
                
                out = model.transformer_decoder(tgt_emb, img_memory)
                logits = model.fc_out(out[:, -1, :])
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                top_k = log_probs.topk(k)
                for i in range(k):
                    idx = top_k.indices[0][i].item()
                    val = top_k.values[0][i].item()
                    if len(seq)>0 and idx == seq[-1]: val -= 10.0
                    next_cands.append((score + val, seq + [idx]))
            
            candidates = sorted(next_cands, key=lambda x: x[0], reverse=True)[:k]
            if all(c[1][-1] == vocab['<end>'] for c in candidates): break

    return [" ".join([idx2word.get(i, '') for i in candidates[0][1] if i not in [0,1,2,3]])]

if __name__ == "__main__":
    # Test
    print(predict_caption("juan-gris_still-life-with-flowers-1912.jpg", emotion="disgust"))