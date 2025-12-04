import os
import pickle
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src/
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")
IMG_DIR = os.path.join(BASE_DIR, "..", "Initial_Artworks_folder")
INPUT_CSV = os.path.join(DATA_DIR, "artemis_10k_sampled.csv")

# Output paths
PROCESSED_DIR = os.path.join(BASE_DIR, "CNN_LSTM", "processed_data")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Embedding Paths (Adjust these filenames if yours differ!)
GLOVE_FILE = os.path.join(DATA_DIR, "embeddings", "glove.6B.200d.txt")
FASTTEXT_FILE = os.path.join(DATA_DIR, "embeddings", "wiki-news-300d-1M.vec") 

VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20

# --- 1. LOAD DATA ---
print("Loading Dataset...")
if not os.path.exists(INPUT_CSV):
    print(f"Error: Could not find {INPUT_CSV}. Please run EDA script first.")
    exit()
    
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} captions.")

# --- 2. TEXT CLEANING & VOCAB ---
print("Building Vocabulary...")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove punctuation
    return text.split()

# Count words
all_words = []
for caption in df['utterance']:
    all_words.extend(clean_text(caption))

word_counts = Counter(all_words)
most_common = word_counts.most_common(VOCAB_SIZE - 4) # Reserve 4 spots for specials

# Create Mappings
vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
for word, _ in most_common:
    vocab[word] = len(vocab)

print(f"Vocabulary Size: {len(vocab)}")
with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "wb") as f:
    pickle.dump(vocab, f)

# --- 3. PREPARE SEQUENCES ---
print("Tokenizing sequences...")
sequences = []
for caption in df['utterance']:
    words = clean_text(caption)
    # Start + Words + End
    seq = [1] + [vocab.get(w, 3) for w in words] + [2]
    
    # Pad or Truncate
    if len(seq) < MAX_SEQ_LEN:
        seq += [0] * (MAX_SEQ_LEN - len(seq))
    else:
        seq = seq[:MAX_SEQ_LEN]
    sequences.append(seq)

# Save processed dataframe with image paths and sequences
df['sequence'] = sequences
# Update image paths to be absolute for training
df['abs_image_path'] = df.apply(lambda row: os.path.join(IMG_DIR, row['art_style'], row['painting'] + '.jpg'), axis=1)
# Filter missing images
df = df[df['abs_image_path'].apply(os.path.exists)]

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

train_df.to_pickle(os.path.join(PROCESSED_DIR, "train_data.pkl"))
val_df.to_pickle(os.path.join(PROCESSED_DIR, "val_data.pkl"))
test_df.to_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
print(f"Splits Saved: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

# --- 4. GENERATE EMBEDDING MATRICES ---

def create_embedding_matrix(name, embedding_dict, dim):
    matrix = np.zeros((len(vocab), dim))
    hits = 0
    for word, i in vocab.items():
        if word in embedding_dict:
            matrix[i] = embedding_dict[word]
            hits += 1
        else:
            # Random init for unknown words
            matrix[i] = np.random.normal(scale=0.6, size=(dim,))
    print(f"{name} Coverage: {hits}/{len(vocab)}")
    return matrix

# A. GloVe
print("\nProcessing GloVe...")
glove_index = {}
if os.path.exists(GLOVE_FILE):
    with open(GLOVE_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                glove_index[word] = np.asarray(values[1:], dtype='float32')
    glove_matrix = create_embedding_matrix("GloVe", glove_index, 200)
    np.save(os.path.join(PROCESSED_DIR, "emb_glove.npy"), glove_matrix)
else:
    print("GloVe file not found. Skipping.")

# B. FastText (Similar parsing to GloVe usually)
print("\nProcessing FastText...")
fasttext_index = {}
if os.path.exists(FASTTEXT_FILE):
    with open(FASTTEXT_FILE, encoding='utf-8') as f:
        f.readline() # Skip header
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                fasttext_index[word] = np.asarray(values[1:], dtype='float32')
    fasttext_matrix = create_embedding_matrix("FastText", fasttext_index, 300)
    np.save(os.path.join(PROCESSED_DIR, "emb_fasttext.npy"), fasttext_matrix)
else:
    print(f"FastText file not found at {FASTTEXT_FILE}. Skipping.")

# C. TF-IDF with PCA (TruncatedSVD)
print("\nProcessing TF-IDF + PCA...")
# We need word vectors, not document vectors.
# Strategy: 
# 1. Compute TF-IDF matrix for the corpus (N_docs x Vocab)
# 2. Transpose to get (Vocab x N_docs) - creates a vector for each word
# 3. Apply SVD to reduce N_docs dimensions down to 200
corpus = [" ".join(clean_text(text)) for text in df['utterance']]
tfidf = TfidfVectorizer(vocabulary=vocab, token_pattern=r"(?u)\b\w+\b")
X = tfidf.fit_transform(corpus)

# Transpose so rows = words
X_T = X.T 
svd = TruncatedSVD(n_components=200, random_state=42)
tfidf_pca_matrix = svd.fit_transform(X_T) # Result: (Vocab_size, 200)

print(f"TF-IDF Matrix Shape: {tfidf_pca_matrix.shape}")
np.save(os.path.join(PROCESSED_DIR, "emb_tfidf_pca.npy"), tfidf_pca_matrix)

print("\nPreprocessing Complete. Files saved to:", PROCESSED_DIR)