import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = os.path.join("..", "Data")
INPUT_CSV = os.path.join(DATA_DIR, "artemis_10k_sampled.csv")
IMG_DIR = os.path.join("..", "Initial_Artworks_folder")

# Hyperparameters from your EDA
VOCAB_SIZE = 8000       # Limit to top 8k words (Assignment asks for 5k-10k)
MAX_SEQ_LEN = 40        # Covers 95%+ of your captions (Avg is 15)
OOV_TOKEN = "<unk>"     # Token for unknown words

# ==========================================
# 1. TEXT CLEANING & PREPARATION
# ==========================================
print("--- 1. Processing Text ---")
df = pd.read_csv(INPUT_CSV)

# Add Start/End Tokens (Critical for Transformer & LSTM)
# We add space padding to ensure tokenization works cleanly
df['processed_caption'] = '<start> ' + df['utterance'].astype(str).str.lower() + ' <end>'

# Initialize Tokenizer
# filters: We keep < and > so <start>/<end> aren't stripped
tokenizer = Tokenizer(
    num_words=VOCAB_SIZE, 
    oov_token=OOV_TOKEN,
    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~' 
)

print("Fitting tokenizer on captions...")
tokenizer.fit_on_texts(df['processed_caption'])

# Save Tokenizer (You will need this for the final Demo/Evaluation)
tokenizer_path = os.path.join(DATA_DIR, "tokenizer.pkl")
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to {tokenizer_path}")

# Convert text to sequences (numbers)
sequences = tokenizer.texts_to_sequences(df['processed_caption'])

# Pad sequences (Post-padding is standard for Transformers)
captions_padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

# Add sequences back to dataframe (store as string to save in CSV)
df['caption_seq'] = list(captions_padded)
df['caption_seq'] = df['caption_seq'].apply(lambda x: str(list(x))) 

# ==========================================
# 2. IMAGE PATH VALIDATION
# ==========================================
print("\n--- 2. Validating Image Paths ---")
# Ensure we point to the correct file path
def get_image_path(row):
    # Logic: Initial_Artworks_folder / Art_Style / Painting.jpg
    return os.path.join(row['art_style'], row['painting'] + '.jpg')

df['image_path'] = df.apply(get_image_path, axis=1)

# Check if they actually exist (sanity check)
# We check the first 100 to save time
valid_count = 0
for idx, row in df.head(100).iterrows():
    full_path = os.path.join(IMG_DIR, row['image_path'])
    if os.path.exists(full_path):
        valid_count += 1

print(f"Path Check: {valid_count}/100 sample images found on disk.")
if valid_count < 80:
    print("WARNING: Many images seem missing. Check IMG_DIR path!")

# ==========================================
# 3. SPLITTING DATA
# ==========================================
print("\n--- 3. Splitting Train/Val/Test ---")
# 80% Train, 10% Val, 10% Test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print(f"Train Size: {len(train_df)}")
print(f"Val Size:   {len(val_df)}")
print(f"Test Size:  {len(test_df)}")

# Save splits
train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(DATA_DIR, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)

print(f"\nSUCCESS: Preprocessing done. Splits saved to {DATA_DIR}")
print(f"Config Used -> Vocab: {VOCAB_SIZE}, Max Len: {MAX_SEQ_LEN}")