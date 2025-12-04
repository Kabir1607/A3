import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import pickle
import re

# ==========================================
# 1. CONFIGURATION & CUSTOM LAYERS
# ==========================================
# We must redefine these classes so Keras knows how to load the model
class Patches(layers.Layer):
    def __init__(self, patch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images, sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID"
        )
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=49, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patch) + self.position_embedding(positions)
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection.units})
        return config

DATA_DIR = os.path.join("..", "Data")
IMG_DIR = os.path.join("..", "Initial_Artworks_folder")
RESULTS_DIR = os.path.join("..", "Results")
TEST_CSV = os.path.join(DATA_DIR, "test_split.csv")
MODEL_PATH = os.path.join(DATA_DIR, "best_transformer_model.keras")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.pkl")
IMAGE_SIZE = 224
MAX_LEN = 40

# ==========================================
# 2. GENERATION FUNCTIONS (FIXED)
# ==========================================
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return img

def generate_caption(model, tokenizer, image_path):
    img = load_image(image_path)
    img = tf.expand_dims(img, axis=0) # Shape: (1, 224, 224, 3)
    
    start_id = tokenizer.word_index.get("<start>")
    end_id = tokenizer.word_index.get("<end>")
    
    output = [start_id]
    
    for _ in range(MAX_LEN):
        # --- FIX: Handle Padding Correctly ---
        # 1. Create 1D tensor
        cap_seq = tf.constant(output) 
        # 2. Calculate padding needed
        pad_len = MAX_LEN - tf.shape(cap_seq)[0]
        # 3. Pad 1D -> 1D
        cap_in = tf.pad(cap_seq, [[0, pad_len]])
        # 4. Expand to 2D (Batch, Seq)
        cap_in = tf.expand_dims(cap_in, 0) 
        # -------------------------------------
        
        preds = model.predict([img, cap_in], verbose=0)
        
        idx = len(output) - 1
        logits = preds[0, idx, :]
        next_id = int(np.argmax(logits))
        
        if next_id == end_id: break
        output.append(next_id)
        
    words = [tokenizer.index_word.get(i, "") for i in output[1:]] 
    return " ".join(words)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load Model with Custom Layers
    model = keras.models.load_model(
        MODEL_PATH, 
        custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder}
    )
    
    # Load Tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
        
    print("Model loaded. Starting evaluation...")
    
    if os.path.exists(TEST_CSV):
        df = pd.read_csv(TEST_CSV)
        # Fix paths
        df["full_path"] = df["image_path"].apply(lambda x: os.path.join(IMG_DIR, str(x)))
        df = df[df["full_path"].apply(os.path.exists)]
        
        # Evaluate on random 10 samples
        samples = df.sample(10)
        results_path = os.path.join(RESULTS_DIR, "final_transformer_examples.txt")
        
        with open(results_path, "w") as f:
            f.write("FINAL TRANSFORMER EVALUATION EXAMPLES\n=====================================\n\n")
            
            for i, row in samples.iterrows():
                print(f"Generating caption for {row['painting']}...")
                pred = generate_caption(model, tokenizer, row["full_path"])
                
                log_entry = f"Style: {row['art_style']}\nEmotion: {row['emotion']}\nRef:  {row['utterance']}\nPred: {pred}\n{'-'*40}\n"
                print(log_entry)
                f.write(log_entry)
                
        print(f"\nSuccess! Examples saved to {results_path}")
    else:
        print("Test CSV not found.")