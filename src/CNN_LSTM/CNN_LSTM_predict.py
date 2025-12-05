import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
import os
import re

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM_Results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_GloVe.pth") # Using your GloVe model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 100
HIDDEN_DIM = 256
MAX_LEN = 20

# --- MODEL ---
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_mat=None, freeze_emb=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # Dummy init if matrix missing (just for loading state_dict)
        if embed_mat is None: embed_mat = torch.zeros(vocab_size, EMBED_DIM)
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

# --- PREDICT ---
# Added 'emotion' argument so it doesn't crash if you pass it
def predict_caption(image_path, emotion=None, k=3):
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f: vocab = pickle.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    
    model = CaptionModel(len(vocab), None).to(DEVICE)
    if not os.path.exists(MODEL_PATH): return ["Model not found!"]
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    try:
        img = Image.open(image_path).convert("RGB")
    except:
        return ["Error loading image"]
        
    img = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        img_feat = model.img_proj(model.cnn(img)).unsqueeze(1)
        candidates = [(0.0, [vocab['<start>']], None)]
        
        for _ in range(MAX_LEN):
            next_cands = []
            for score, seq, states in candidates:
                if seq[-1] == vocab['<end>']:
                    next_cands.append((score, seq, states))
                    continue
                
                if len(seq) == 1: inputs = img_feat
                else: inputs = model.embed(torch.tensor([[seq[-1]]]).to(DEVICE))
                
                if states is None: out, states = model.lstm(inputs)
                else: out, states = model.lstm(inputs, states)
                
                logits = model.fc(out).squeeze()
                probs = torch.nn.functional.log_softmax(logits, dim=0)
                top_k = probs.topk(k)
                
                for i in range(k):
                    idx = top_k.indices[i].item()
                    val = top_k.values[i].item()
                    if len(seq)>1 and idx == seq[-1]: val -= 5.0 # Repetition penalty
                    next_cands.append((score + val, seq + [idx], states))
            
            candidates = sorted(next_cands, key=lambda x: x[0], reverse=True)[:k]
            if all(c[1][-1] == vocab['<end>'] for c in candidates): break

    return [" ".join([idx2word.get(i, '') for i in candidates[0][1] if i not in [0,1,2,3]])]

if __name__ == "__main__":
    # Test
    print(predict_caption("paolo-veronese_the-consecration-of-saint-nicholas.jpg", emotion="sadness"))