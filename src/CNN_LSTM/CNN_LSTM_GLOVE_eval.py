import torch
import torch.nn as nn
from torch.utils.data import Dataset
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
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_GloVe.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 100 
HIDDEN_DIM = 256
MAX_LEN = 20
BEAM_WIDTH = 3  # Top 3 paths

# --- MODEL (Must match Training) ---
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_mat, freeze_emb=True):
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

# --- BEAM SEARCH GENERATION ---
def generate_caption_beam(model, image, vocab, idx2word, max_len=20, k=3):
    model.eval()
    with torch.no_grad():
        img = image.unsqueeze(0).to(DEVICE)
        img_feat = model.img_proj(model.cnn(img)).unsqueeze(1)
        
        # Tuple: (score, current_seq, (hidden, cell))
        # We effectively perform "batch size = k" operations
        candidates = [(0.0, [vocab['<start>']], None)]
        
        for _ in range(max_len):
            all_next_candidates = []
            
            for score, seq, states in candidates:
                if seq[-1] == vocab['<end>']:
                    all_next_candidates.append((score, seq, states))
                    continue
                
                # Prepare input
                if len(seq) == 1:
                    # First step: Image Feature
                    inputs = img_feat
                    # LSTM step (image -> hidden)
                    out, (h, c) = model.lstm(inputs, None)
                else:
                    # Subsequent steps: Word Embedding
                    word_idx = seq[-1]
                    word_tensor = torch.tensor([[word_idx]]).to(DEVICE)
                    inputs = model.embed(word_tensor)
                    out, (h, c) = model.lstm(inputs, states)

                # Prediction
                output = model.fc(out).squeeze(0).squeeze(0) # (Vocab_Size)
                log_probs = torch.nn.functional.log_softmax(output, dim=0)
                
                # Get top k words
                top_k_probs, top_k_ids = log_probs.topk(k)
                
                for i in range(k):
                    word_idx = top_k_ids[i].item()
                    added_score = top_k_probs[i].item()
                    
                    # Repetition Penalty (Soft)
                    if len(seq) > 2 and word_idx == seq[-1]:
                         added_score -= 10.0

                    all_next_candidates.append((score + added_score, seq + [word_idx], (h, c)))
            
            # Sort by score and keep top k
            candidates = sorted(all_next_candidates, key=lambda x: x[0], reverse=True)[:k]
            
            # Early stop if all candidates ended
            if all(c[1][-1] == vocab['<end>'] for c in candidates):
                break
                
        # Return best sequence
        best_seq = candidates[0][1]
        words = [idx2word.get(idx, '') for idx in best_seq if idx not in [vocab['<start>'], vocab['<end>']]]
        return " ".join(words)

# --- MAIN ---
if __name__ == "__main__":
    print(f"--- Evaluating GloVe with BEAM SEARCH (k={BEAM_WIDTH}) ---")
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f: vocab = pickle.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    
    test_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    # Load Model
    dummy_mat = np.zeros((len(vocab), EMBED_DIM))
    model = CaptionModel(len(vocab), dummy_mat).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Model not found! Run Training first.")
        exit()
    
    print("\n=== Sample Captions ===")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    samples = test_df.sample(5)
    for i, row in samples.iterrows():
        if not os.path.exists(row['abs_path']): continue
        img = Image.open(row['abs_path']).convert("RGB")
        img_t = transform(img)
        
        pred = generate_caption_beam(model, img_t, vocab, idx2word, k=BEAM_WIDTH)
        print(f"Ref: {row['utterance']}")
        print(f"Pred: {pred}")
        print("-" * 30)

    print("\n=== Calculating BLEU ===")
    scores = []
    # Subset for speed
    for i, row in test_df.head(200).iterrows():
        if not os.path.exists(row['abs_path']): continue
        img = Image.open(row['abs_path']).convert("RGB")
        img_t = transform(img)
        pred = generate_caption_beam(model, img_t, vocab, idx2word, k=BEAM_WIDTH).split()
        ref = re.sub(r'[^\w\s]', '', str(row['utterance']).lower()).split()
        
        try: score = sentence_bleu([ref], pred, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        except: score = 0
        scores.append(score)
        
    print(f"Average BLEU-1: {np.mean(scores):.4f}")