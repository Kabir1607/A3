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

# --- 1. PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A. FIND IMAGES
possible_data_paths = [
    os.path.join(SCRIPT_DIR, "Initial_Artworks_folder"),
    os.path.join(PROJECT_ROOT, "Initial_Artworks_folder"),
    os.path.join(PROJECT_ROOT, "Data", "wikiart"),
    os.path.join(SCRIPT_DIR, "wikiart")
]
DATA_ROOT = next((p for p in possible_data_paths if os.path.exists(p)), None)

if DATA_ROOT is None:
    print("⚠️  WARNING: Could not find 'Initial_Artworks_folder'. Creating empty placeholder.")
    DATA_ROOT = os.path.join(SCRIPT_DIR, "Initial_Artworks_folder")
    os.makedirs(DATA_ROOT, exist_ok=True)
else:
    print(f"✅ Found Images at: {DATA_ROOT}")

# B. FIND ANNOTATIONS (OPTIMIZED)
# Check for the SAMPLED 10k file first to speed up processing
possible_csv_paths = [
    os.path.join(PROJECT_ROOT, "Data", "artemis_10k_sampled.csv"), # <--- Priority 1
    os.path.join(SCRIPT_DIR, "artemis_10k_sampled.csv"),           # <--- Priority 2
    os.path.join(PROJECT_ROOT, "Data", "artemis_dataset_release_v0.csv") # Fallback (Slow)
]
ANNOTATIONS_PATH = next((p for p in possible_csv_paths if os.path.exists(p)), None)
print(f"✅ Using Annotations: {ANNOTATIONS_PATH}")

# C. FIND EMBEDDINGS
possible_glove = [
    os.path.join(SCRIPT_DIR, "glove.6B.200d.txt"),
    os.path.join(PROJECT_ROOT, "Data", "embeddings", "glove.6B.200d.txt")
]
GLOVE_PATH = next((p for p in possible_glove if os.path.exists(p)), None)

possible_fasttext = [
    os.path.join(SCRIPT_DIR, "wiki-news-300d-1M.vec"),
    os.path.join(PROJECT_ROOT, "Data", "embeddings", "wiki-news-300d-1M.vec")
]
FASTTEXT_PATH = next((p for p in possible_fasttext if os.path.exists(p)), None)

VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20

# --- 2. IMAGE SPLIT ---
print("\n--- 1. Processing Images ---")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

try:
    dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    splits = {'train': train_idx, 'val': val_idx, 'test': test_idx}
    with open(os.path.join(OUTPUT_DIR, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
    print(f"Splits Created: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
except Exception as e:
    print(f"⚠️  Skipping Image processing (Data structure issue?): {e}")
    # Create dummy splits
    splits = {'train': [], 'val': [], 'test': []}
    with open(os.path.join(OUTPUT_DIR, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)

# --- 3. TEXT PREPROCESSING ---
print("\n--- 2. Processing Text ---")
if ANNOTATIONS_PATH:
    df = pd.read_csv(ANNOTATIONS_PATH)
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    all_tokens = []
    for caption in df['utterance']:
        all_tokens.extend(clean_text(caption))

    counter = Counter(all_tokens)
    most_common = counter.most_common(VOCAB_SIZE - 4)
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    for token, _ in most_common:
        vocab[token] = len(vocab)

    with open(os.path.join(OUTPUT_DIR, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary Built: {len(vocab)} tokens")

    # --- 4. EMBEDDINGS ---
    def load_vectors(fname, vocab, dim):
        print(f"Loading {os.path.basename(fname)}...")
        matrix = np.zeros((len(vocab), dim))
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            # FastText check
            first_line = f.readline().split()
            if len(first_line) != dim + 1:
                 f.seek(0) 

            for line in f:
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if word in vocab:
                    try: matrix[vocab[word]] = np.array(tokens[1:], dtype=np.float32)
                    except: continue
        return matrix

    # A. GloVe
    if GLOVE_PATH:
        glove_matrix = load_vectors(GLOVE_PATH, vocab, 200)
        np.save(os.path.join(OUTPUT_DIR, 'emb_glove.npy'), glove_matrix)
    else:
        print("⚠️  GloVe file not found. Generating random placeholder.")
        np.save(os.path.join(OUTPUT_DIR, 'emb_glove.npy'), np.random.normal(0, 1, (len(vocab), 200)))

    # B. FastText
    if FASTTEXT_PATH:
        fasttext_matrix = load_vectors(FASTTEXT_PATH, vocab, 300)
        np.save(os.path.join(OUTPUT_DIR, 'emb_fasttext.npy'), fasttext_matrix)
    else:
        print("⚠️  FastText file not found. Generating random placeholder.")
        np.save(os.path.join(OUTPUT_DIR, 'emb_fasttext.npy'), np.random.normal(0, 1, (len(vocab), 300)))

    # C. TF-IDF + PCA
    print("Calculating TF-IDF + PCA (Sampled Data)...")
    corpus = [" ".join(clean_text(u)) for u in df['utterance']]
    vectorizer = TfidfVectorizer(vocabulary=vocab, token_pattern=r"\b\w+\b")
    X = vectorizer.fit_transform(corpus)
    
    svd = TruncatedSVD(n_components=200, random_state=42)
    svd.fit(X) 
    tfidf_matrix = svd.components_.T 
    
    final_tfidf = np.zeros((len(vocab), 200))
    for word, idx in vocab.items():
        if word in vectorizer.vocabulary_:
            final_tfidf[idx] = tfidf_matrix[vectorizer.vocabulary_[word]]
            
    np.save(os.path.join(OUTPUT_DIR, 'emb_tfidf.npy'), final_tfidf)

print("\nPreprocessing Complete.")