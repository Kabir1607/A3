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
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model_GloVe.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 100 
HIDDEN_DIM = 256
MAX_LEN = 20

# --- MODEL ---
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

# --- GENERATE WITH TEMPERATURE ---
def generate_caption(model, image, vocab, idx2word, max_len=20, temperature=0.8):
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
            
            # For the first step, we technically used image. 
            # For simplicity in this loop structure, we just feed the previous word embedding
            # The LSTM state carries the image info from step 0 (handled outside loop in pure implementation, 
            # but here we just chain inputs. Correct seq is Image -> Word -> Word)
            
            if len(result) == 0:
                # First step: Image is already processed into hidden state? 
                # Our model structure is concat. Let's do step-by-step manually:
                out, states = model.lstm(inputs, states)
            else:
                out, states = model.lstm(word_emb, states)

            output = model.fc(out) # (1, 1, Vocab)
            logits = output[0, 0, :] / temperature # Apply Temp
            
            # Block <unk> and repetition
            logits[vocab['<unk>']] = -float('inf')
            if len(result) > 0:
                # Penalize previous word to prevent "very very very"
                prev_idx = vocab.get(result[-1], 0)
                logits[prev_idx] -= 3.0 

            # Sample from distribution
            probs = torch.nn.functional.softmax(logits, dim=0)
            next_word_idx = torch.multinomial(probs, 1).item()
            
            if next_word_idx == vocab['<end>']: break
            
            word = idx2word.get(next_word_idx, '')
            result.append(word)
            
    return " ".join(result)

# --- MAIN ---
if __name__ == "__main__":
    print(f"--- Evaluating GloVe (Temperature Sampling) ---")
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f: vocab = pickle.load(f)
    idx2word = {v: k for k, v in vocab.items()}
    
    test_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "test.pkl"))
    
    # Dummy matrix for shape
    dummy_mat = np.zeros((len(vocab), EMBED_DIM))
    model = CaptionModel(len(vocab), dummy_mat).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Model not found! Using random weights (Expect bad results)")
    
    model.eval()
    
    print("\n=== Sample Captions ===")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    samples = test_df.sample(5)
    for i, row in samples.iterrows():
        if not os.path.exists(row['abs_path']): continue
        img = Image.open(row['abs_path']).convert("RGB")
        img_t = transform(img)
        
        # Try temperature 0.7 or 0.8 for best balance
        pred = generate_caption(model, img_t, vocab, idx2word, temperature=0.8)
        print(f"Ref: {row['utterance']}")
        print(f"Pred: {pred}")
        print("-" * 30)

    # Scoring
    print("\n=== Calculating BLEU ===")
    scores = []
    for i, row in test_df.head(200).iterrows():
        if not os.path.exists(row['abs_path']): continue
        img = Image.open(row['abs_path']).convert("RGB")
        img_t = transform(img)
        
        # Use lower temp for scoring to be more accurate/safe
        pred = generate_caption(model, img_t, vocab, idx2word, temperature=0.1).split()
        ref = re.sub(r'[^\w\s]', '', str(row['utterance']).lower()).split()
        
        try: score = sentence_bleu([ref], pred, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        except: score = 0
        scores.append(score)
        
    print(f"Average BLEU-1: {np.mean(scores):.4f}")