import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from torchvision import datasets, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. PATHS
DATA_ROOT = None
possible_img_paths = [
    os.path.join(SCRIPT_DIR, "Initial_Artworks_folder"),
    os.path.join(PROJECT_ROOT, "Initial_Artworks_folder")
]
DATA_ROOT = next((p for p in possible_img_paths if os.path.exists(p)), None)

if DATA_ROOT is None:
    print("⚠️  WARNING: Image folder not found. Creating placeholder.")
    DATA_ROOT = os.path.join(SCRIPT_DIR, "Initial_Artworks_folder")
    os.makedirs(DATA_ROOT, exist_ok=True)
else:
    print(f"✅ Images found at: {DATA_ROOT}")

# 2. ANNOTATIONS (Sampled)
CSV_PATH = next((p for p in [
    os.path.join(PROJECT_ROOT, "Data", "artemis_10k_sampled.csv"),
    os.path.join(SCRIPT_DIR, "artemis_10k_sampled.csv")
] if os.path.exists(p)), None)

# 3. GLOVE (The Real Embedding)
GLOVE_PATH = next((p for p in [
    os.path.join(SCRIPT_DIR, "glove.6B.100d.txt"),
    os.path.join(PROJECT_ROOT, "Data", "embeddings", "glove.6B.100d.txt")
] if os.path.exists(p)), None)

VOCAB_SIZE = 5000
MAX_LEN = 20

# --- PROCESSING ---
print("\n--- 1. Processing Text & Splits ---")
if CSV_PATH:
    df = pd.read_csv(CSV_PATH)
    
    # Clean & Tokenize
    def clean(text):
        return re.sub(r'[^\w\s]', '', str(text).lower()).split()

    all_tokens = [w for t in df['utterance'] for w in clean(t)]
    counts = Counter(all_tokens).most_common(VOCAB_SIZE - 4)
    vocab = {'<pad>':0, '<start>':1, '<end>':2, '<unk>':3}
    for w, _ in counts: vocab[w] = len(vocab)

    with open(os.path.join(OUTPUT_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    
    # Tokenize
    def tokenize(text):
        seq = [1] + [vocab.get(w, 3) for w in clean(text)] + [2]
        if len(seq) < MAX_LEN: seq += [0]*(MAX_LEN - len(seq))
        else: seq = seq[:MAX_LEN]
        return seq

    df['sequence'] = df['utterance'].apply(tokenize)
    
    # Path Fix
    def get_path(row):
        path = os.path.join(DATA_ROOT, row['art_style'], row['painting'] + '.jpg')
        return path if os.path.exists(path) else None
    
    df['abs_path'] = df.apply(get_path, axis=1)
    df = df.dropna(subset=['abs_path'])
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42) # 10% Val, 10% Test
    
    train_df.to_pickle(os.path.join(OUTPUT_DIR, "train.pkl"))
    val_df.to_pickle(os.path.join(OUTPUT_DIR, "val.pkl"))
    test_df.to_pickle(os.path.join(OUTPUT_DIR, "test.pkl"))
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # --- 2. EMBEDDINGS ---
    print("\n--- 2. Generating Embeddings ---")
    
    # A. GloVe (Real)
    glove_mat = np.zeros((len(vocab), 100))
    if GLOVE_PATH:
        print(f"Loading GloVe from {os.path.basename(GLOVE_PATH)}...")
        with open(GLOVE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                p = line.split()
                if p[0] in vocab:
                    try: glove_mat[vocab[p[0]]] = np.array(p[1:], dtype=np.float32)
                    except: pass
    else:
        print("⚠️ GloVe not found. Using Random (Baseline).")
        glove_mat = np.random.normal(0, 1, (len(vocab), 100))
    np.save(os.path.join(OUTPUT_DIR, "emb_glove.npy"), glove_mat)

    # B. TF-IDF (Statistical)
    print("Calculating TF-IDF...")
    vec = TfidfVectorizer(vocabulary=vocab)
    X = vec.fit_transform(df['utterance'])
    svd = TruncatedSVD(n_components=100) # Match GloVe dim
    tfidf_mat = svd.fit_transform(X.T)
    
    final_tfidf = np.zeros((len(vocab), 100))
    for w, i in vocab.items():
        if w in vec.vocabulary_:
            final_tfidf[i] = tfidf_mat[vec.vocabulary_[w]]
    np.save(os.path.join(OUTPUT_DIR, "emb_tfidf.npy"), final_tfidf)
    
    # C. Learned (Random Baseline)
    # We create a random matrix to represent "No Pre-training"
    learned_mat = np.random.normal(0, 1, (len(vocab), 100))
    np.save(os.path.join(OUTPUT_DIR, "emb_learned.npy"), learned_mat)

print("Preprocessing Done.")