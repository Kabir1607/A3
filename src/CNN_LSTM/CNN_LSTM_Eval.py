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

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM_Results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_cnn_lstm.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (Must match Training Script)
EMBED_DIM = 100  # Matches GloVe 100d
HIDDEN_DIM = 256
MAX_LEN = 20

# --- MODEL ARCHITECTURE ---
# Must match the class definition in CNN_LSTM_Model.py exactly
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_mat, freeze_emb=False):
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
        # We initialize with a dummy tensor; state_dict will overwrite weights
        if embed_mat is None:
            # Fallback if matrix missing, shape must be correct
            embed_mat = torch.zeros(vocab_size, EMBED_DIM)
        
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

# --- GENERATION FUNCTION ---
def generate_caption(model, image, vocab, idx2word, max_len=20):
    model.eval()
    with torch.no_grad():
        # Encode Image
        # Pass through CNN parts manually to get image feature
        img = image.unsqueeze(0).to(DEVICE)
        img_feat = model.cnn(img)           # (1, Hidden)
        img_feat = model.img_proj(img_feat) # (1, Embed)
        img_feat = img_feat.unsqueeze(1)    # (1, 1, Embed)
        
        # Start Token
        inputs = img_feat
        states = None
        result = []
        
        # First LSTM step (Image)
        out, states = model.lstm(inputs, states)
        
        # Start generating words
        next_word_idx = vocab['<start>']
        
        for _ in range(max_len):
            # Embed current word
            word_tensor = torch.tensor([[next_word_idx]]).to(DEVICE)
            word_emb = model.embed(word_tensor) # (1, 1, Embed)
            
            # LSTM step
            out, states = model.lstm(word_emb, states)
            output = model.fc(out) # (1, 1, Vocab)
            
            # Greedy prediction
            next_word_idx = output.argmax(2).item()
            
            if next_word_idx == vocab['<end>']:
                break
                
            word = idx2word.get(next_word_idx, '<unk>')
            result.append(word)
            
    return " ".join(result)

# --- MAIN EVALUATION ---
if __name__ == "__main__":
    print(f"--- Evaluating CNN+LSTM Model ---")
    
    # 1. Load Resources
    print("Loading Vocab and Data...")
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    
    # Load Test Data
    test_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    # Load Embedding Matrix (needed for model init shape)
    # We try GloVe first as that was likely the best model
    try:
        embed_matrix = np.load(os.path.join(PROCESSED_DIR, "emb_glove.npy"))
    except:
        print("Warning: Embedding matrix not found. Using random init for shape placeholder.")
        embed_matrix = np.zeros((len(vocab), EMBED_DIM))

    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    model = CaptionModel(len(vocab), embed_matrix).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
        print("Did you run CNN_LSTM_Model.py?")
        exit()
        
    model.eval()
    
    # 3. Qualitative: 5 Random Samples
    print("\n=== Sample Captions ===")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    samples = test_df.sample(5)
    for i, row in samples.iterrows():
        try:
            # Handle path (robust check)
            if os.path.exists(row['abs_path']):
                img_path = row['abs_path']
            else:
                continue

            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            
            pred = generate_caption(model, img_tensor, vocab, idx2word)
            print(f"Ref: {row['utterance']}")
            print(f"Pred: {pred}")
            print("-" * 30)
        except Exception as e:
            print(f"Error on sample: {e}")

    # 4. Quantitative: BLEU Score
    print("\n=== Calculating BLEU Score (Subset of 200) ===")
    scores = []
    subset = test_df.head(200)
    
    for i, row in subset.iterrows():
        try:
            if not os.path.exists(row['abs_path']): continue
            
            img = Image.open(row['abs_path']).convert("RGB")
            img_tensor = transform(img)
            
            pred_tokens = generate_caption(model, img_tensor, vocab, idx2word).split()
            
            # Reference tokens (cleaned)
            ref_tokens = re.sub(r'[^\w\s]', '', str(row['utterance']).lower()).split()
            
            score = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
            scores.append(score)
        except: pass
        
    avg_bleu = np.mean(scores)
    print(f"Average BLEU-1 Score: {avg_bleu:.4f}")
    
    # Save Report
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
        f.write("CNN + LSTM EVALUATION\n")
        f.write(f"BLEU-1 Score: {avg_bleu:.4f}\n")
    print(f"Saved to {RESULTS_DIR}/evaluation_report.txt")