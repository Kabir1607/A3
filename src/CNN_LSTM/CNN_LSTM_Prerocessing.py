import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_ROOT = "data/wikiart"
ANNOTATIONS_PATH = "data/annotations/artemis_dataset_release_v0.csv"
OUTPUT_DIR = "processed_data"
GLOVE_PATH = "data/embeddings/glove.6B.200d.txt"
FASTTEXT_PATH = "data/embeddings/wiki-news-300d-1M.vec"  # Update filename if needed
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. DATA LOAD & SPLIT ---
print("Loading and Splitting Data...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize to [0, 1] is implicit in ToTensor, but usually we standardize 
    # to ImageNet means/stds for CNNs. Keeping it simple [0,1] as per assignment text.
])

# Use your partner's logic for consistency, but adding Test split
dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
indices = list(range(len(dataset)))

# Stratified split (simplified for brevity, assuming class balance logic was done)
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# Save splits
splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
with open(f'{OUTPUT_DIR}/splits.pkl', 'wb') as f:
    pickle.dump(splits, f)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

# --- 2. TEXT PREPROCESSING ---
print("Building Vocabulary...")
df = pd.read_csv(ANNOTATIONS_PATH)

# Clean Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

# Build Vocab
all_tokens = []
for caption in df['utterance']:
    all_tokens.extend(clean_text(caption))

counter = Counter(all_tokens)
most_common = counter.most_common(VOCAB_SIZE - 4) # Reserve 4 spots
vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
for token, _ in most_common:
    vocab[token] = len(vocab)

with open(f'{OUTPUT_DIR}/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# --- 3. EMBEDDING MATRICES ---

def load_vectors(fname, vocab, dim):
    print(f"Loading embeddings from {fname}...")
    matrix = np.zeros((len(vocab), dim))
    if not os.path.exists(fname):
        print(f"WARNING: {fname} not found. Initializing random.")
        return np.random.normal(scale=0.6, size=(len(vocab), dim))
        
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for line in f:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in vocab:
                matrix[vocab[word]] = np.array(tokens[1:], dtype=np.float32)
    return matrix

# A. GloVe
glove_matrix = load_vectors(GLOVE_PATH, vocab, 200)
np.save(f'{OUTPUT_DIR}/emb_glove.npy', glove_matrix)

# B. FastText
# Note: FastText files usually have a header line to skip
fasttext_matrix = np.zeros((len(vocab), 300))
if os.path.exists(FASTTEXT_PATH):
    print("Loading FastText...")
    with open(FASTTEXT_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f) # Skip header
        for line in f:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in vocab:
                fasttext_matrix[vocab[word]] = np.array(tokens[1:], dtype=np.float32)
np.save(f'{OUTPUT_DIR}/emb_fasttext.npy', fasttext_matrix)

# C. TF-IDF (Word Level via SVD components)
print("Generating TF-IDF Word Embeddings...")
corpus = [" ".join(clean_text(u)) for u in df['utterance']]
vectorizer = TfidfVectorizer(vocabulary=vocab, token_pattern=r"\b\w+\b")
X = vectorizer.fit_transform(corpus)

# SVD on the Transpose (Terms x Documents) or just use components from Doc x Terms
# The 'components_' attribute of TruncatedSVD on (Doc x Terms) gives (Components x Terms).
# Transposing it gives (Terms x Components), which is a vector for each word.
svd = TruncatedSVD(n_components=200, random_state=42)
svd.fit(X)
# Shape: (Vocab_Size, 200)
tfidf_matrix = svd.components_.T 
# Re-order to match our vocab dictionary explicitly
final_tfidf = np.zeros((len(vocab), 200))
for word, idx in vocab.items():
    if word in vectorizer.vocabulary_:
        # Map sklearn vocab index to our vocab index
        sklearn_idx = vectorizer.vocabulary_[word]
        final_tfidf[idx] = tfidf_matrix[sklearn_idx]

np.save(f'{OUTPUT_DIR}/emb_tfidf.npy', final_tfidf)

print("Preprocessing Complete!")