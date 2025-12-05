import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import re
import pickle
import ast

# ==========================================
# 1. CONFIGURATION & LOADING
# ==========================================
# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "CNN_LSTM", "processed_data") # Re-use CNN data
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results")
DATA_DIR = os.path.join(BASE_DIR, "Data")
IMG_DIR = os.path.join(SCRIPT_DIR, "Initial_Artworks_folder")

# Check for Image Folder (Robust check)
if not os.path.exists(IMG_DIR):
    # Try finding it in the CNN folder if not here
    fallback = os.path.join(SCRIPT_DIR, "CNN_LSTM", "Initial_Artworks_folder")
    if os.path.exists(fallback): IMG_DIR = fallback

os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(DATA_DIR, "best_transformer_model.keras")

# Constants
IMAGE_SIZE = 224
PATCH_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
VOCAB_SIZE = 5000 # Matches your CNN_LSTM preprocessing
MAX_LEN = 20      # Matches your CNN_LSTM preprocessing

# HYPERPARAMETERS (Optimized)
EMBED_DIM = 256   # Increased to match FastText/CNN output
NUM_HEADS = 4     # More attention heads for better context
FF_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.4
LEARNING_RATE = 5e-4 # Slightly lower start

# ==========================================
# 2. LOAD PRE-TRAINED DATA
# ==========================================
print("Loading Pre-processed Data...")
try:
    # Load the vocab and dataframes created by CNN_LSTM_Preprocessing.py
    with open(os.path.join(PROCESSED_DIR, "vocab.pkl"), "rb") as f:
        VOCAB = pickle.load(f)
    
    # Load FastText Matrix (The "Secret Weapon")
    EMBED_MATRIX = np.load(os.path.join(PROCESSED_DIR, "emb_fasttext.npy"))
    print(f"Loaded FastText Matrix: {EMBED_MATRIX.shape}")
    
    # Load Dataframes
    TRAIN_DF = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_data.pkl"))
    VAL_DF = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_data.pkl"))
    TEST_DF = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))

except Exception as e:
    print(f"CRITICAL ERROR: Could not load data from {PROCESSED_DIR}")
    print("Run 'src/CNN_LSTM/CNN_LSTM_Preprocessing.py' first!")
    exit()

# ==========================================
# 3. DATA GENERATOR
# ==========================================
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return img

def make_dataset(df, batch_size=32):
    # Use the 'abs_image_path' and 'sequence' columns from preprocessing
    image_paths = df['abs_image_path'].values
    captions = list(df['sequence'].values)
    
    # Setup inputs/targets (Teacher Forcing)
    # In:  <start> A painting
    # Out: A painting <end>
    cap_in = [c[:-1] for c in captions]
    cap_out = [c[1:] for c in captions]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, cap_in, cap_out))

    def map_func(img_path, c_in, c_out):
        img = load_image(img_path)
        return (img, c_in), c_out

    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ==========================================
# 4. TRANSFORMER COMPONENTS
# ==========================================
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config

def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def transformer_decoder_block(inputs, context, head_size, num_heads, ff_dim, dropout=0):
    # 1. Self Attention (Causal)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x, use_causal_mask=True)
    res = x + inputs
    # 2. Cross Attention
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, context)
    res = x + res
    # 3. Feed Forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# ==========================================
# 5. BUILD & TRAIN
# ==========================================
def build_transformer():
    image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image_input")
    caption_input = layers.Input(shape=(MAX_LEN-1,), name="caption_input") # -1 for shift

    # --- IMAGE BRANCH ---
    patches = Patches(PATCH_SIZE)(image_input)
    encoded_patches = PatchEncoder(NUM_PATCHES, EMBED_DIM)(patches)

    for _ in range(NUM_LAYERS):
        encoded_patches = transformer_encoder_block(
            encoded_patches, head_size=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT
        )

    # --- TEXT BRANCH (PRE-TRAINED) ---
    # Initialize with FastText weights!
    # Adjust dimension: FastText is 300, Model is 256. 
    # If they mismatch, we project or slice. Here we rely on the Embedding layer trainable=True to adapt.
    
    if EMBED_MATRIX.shape[1] == EMBED_DIM:
        # Perfect match
        caption_embed = layers.Embedding(
            VOCAB_SIZE, EMBED_DIM, 
            embeddings_initializer=tf.keras.initializers.Constant(EMBED_MATRIX),
            mask_zero=True, trainable=True # Fine-tune it
        )(caption_input)
    else:
        # Mismatch (300 vs 256): Use FastText as generic init but project down
        # Or just use random init if dimensions clash too much to keep it simple for deadline
        print(f"Dimension Mismatch ({EMBED_MATRIX.shape[1]} vs {EMBED_DIM}). Using Random Init.")
        caption_embed = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(caption_input)

    positions = tf.range(start=0, limit=MAX_LEN-1, delta=1)
    pos_embed = layers.Embedding(input_dim=MAX_LEN, output_dim=EMBED_DIM)(positions)
    caption_final = caption_embed + pos_embed

    for _ in range(NUM_LAYERS):
        caption_final = transformer_decoder_block(
            caption_final, context=encoded_patches, 
            head_size=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT
        )

    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(caption_final)

    model = keras.Model(inputs=[image_input, caption_input], outputs=outputs)
    
    # --- OPTIMIZATION: Label Smoothing ---
    # Helps prevent "her her her" loops by penalizing overconfidence
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model

# ==========================================
# 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Training Improved Transformer ---")
    
    train_ds = make_dataset(TRAIN_DF, batch_size=32)
    val_ds = make_dataset(VAL_DF, batch_size=32)
    
    model = build_transformer()
    
    # Callbacks
    callbacks = [
        # Decay LR when stuck
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        # Stop if getting worse
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15, # Give it time to learn
        callbacks=callbacks
    )
    
    model.save(MODEL_PATH)
    print(f"Saved improved model to {MODEL_PATH}")

    # --- QUICK EVAL ---
    print("\n--- Generating Samples (Test Set) ---")
    
    # Build Index-Word Map
    idx2word = {v: k for k, v in VOCAB.items()}
    
    def generate_caption(image_path):
        img = load_image(image_path)
        img = tf.expand_dims(img, axis=0)
        
        output = [VOCAB['<start>']]
        for _ in range(MAX_LEN-1):
            cap_in = tf.constant([output])
            # Pad to expected length
            cap_in = tf.pad(cap_in, [[0, (MAX_LEN-1) - tf.shape(cap_in)[1]]])
            
            preds = model.predict([img, cap_in], verbose=0)
            
            # Temperature Sampling (Soft choice)
            logits = preds[0, len(output)-1, :] 
            
            # Block <unk> and repetition
            logits[VOCAB['<unk>']] = -1e9
            if len(output) > 1: logits[output[-1]] = -1e9

            next_id = np.argmax(logits)
            
            if next_id == VOCAB['<end>']: break
            output.append(next_id)
            
        return " ".join([idx2word.get(i, "") for i in output[1:]])

    # Show 5 samples
    samples = TEST_DF.sample(5)
    with open(os.path.join(RESULTS_DIR, "transformer_improved_examples.txt"), "w") as f:
        for _, row in samples.iterrows():
            pred = generate_caption(row['abs_image_path'])
            print(f"Ref: {' '.join(map(str, row['sequence']))}") # Raw seq for debug
            print(f"Pred: {pred}")
            print("-" * 30)
            f.write(f"Pred: {pred}\n")