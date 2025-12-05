import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import pickle
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM_Results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_GloVe.pth") # Load the GloVe model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 100 
HIDDEN_DIM = 256
MAX_LEN = 20

# --- MODEL (Must Match Training) ---
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_mat, freeze_emb=False):
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
        # Init dummy embedding, will load state_dict over it
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

# --- GENERATE ---
def generate_caption(model, image, vocab, idx2word, max_len=20):
    model.eval()
    with torch.no_grad():
        img = image.unsqueeze(0).to(DEVICE)
        img_feat = model.img_proj(model.cnn(img)).unsqueeze(1)
        
        inputs = img_feat
        states = None
        result = []
        
        next_word_idx = vocab['<start>']
        
        for _ in range(max_len):
            word_tensor = torch.tensor([[next_word_idx]]).to(DEVICE)
            word_emb = model.embed(word_tensor)
            
            out, states = model.lstm(inputs, states) # Feed image first, then words
            # On first step inputs is image. Subsequent steps need word embedding.
            if len(result) == 0: inputs = word_emb
            else: inputs = word_emb # Logic correction for loop
            
            # Actually standard inference loop:
            # 1. Feed Image -> Hidden State
            # 2. Feed Start Token + Hidden State -> First Word
            # 3. Loop
            # But our model concat structure is simple. Let's stick to simple greedy:
            
            output = model.fc(out)
            next_word_idx = output.argmax(2).item()
            
            if next_word_idx == vocab['<end>']: break
            
            word = idx2word.get(next_word_idx, '<unk>')
            result.append(word)
            
            # Update input for next step to be the predicted word
            inputs = model.embed(torch.tensor([[next_word_idx]]).to(DEVICE))

    return " ".join(result)

# --- EVALUATE ---
if __name__ == "__main__":
    print(f"--- Evaluating GloVe Model ---")
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f: vocab = pickle.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    
    test_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    # Load Model
    # Pass dummy matrix for init
    dummy_mat = np.zeros((len(vocab), EMBED_DIM))
    model = CaptionModel(len(vocab), dummy_mat).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("\n=== Sample Captions ===")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    samples = test_df.sample(5)
    for i, row in samples.iterrows():
        if not os.path.exists(row['abs_path']): continue
        img = Image.open(row['abs_path']).convert("RGB")
        img_t = transform(img)
        
        pred = generate_caption(model, img_t, vocab, idx2word)
        print(f"Ref: {row['utterance']}")
        print(f"Pred: {pred}")
        print("-" * 30)

    # BLEU
    print("\n=== Calculating BLEU ===")
    scores = []
    for i, row in test_df.head(200).iterrows():
        if not os.path.exists(row['abs_path']): continue
        img = Image.open(row['abs_path']).convert("RGB")
        img_t = transform(img)
        pred = generate_caption(model, img_t, vocab, idx2word).split()
        ref = re.sub(r'[^\w\s]', '', str(row['utterance']).lower()).split()
        
        try: score = sentence_bleu([ref], pred, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        except: score = 0
        scores.append(score)
        
    print(f"Average BLEU-1: {np.mean(scores):.4f}")