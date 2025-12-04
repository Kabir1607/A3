import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import ast
import re
import pickle
import math

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = os.path.join("..", "Data")
IMG_DIR = os.path.join("..", "Initial_Artworks_folder")
RESULTS_DIR = os.path.join("..", "Results")
TRAIN_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_split.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_split.csv")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "best_transformer_model.keras")

os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 224
PATCH_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
VOCAB_SIZE = 8001
MAX_LEN = 40

# --- HYPERPARAMETERS ---
EMBED_DIM = 64
NUM_HEADS = 2
FF_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.4
LEARNING_RATE = 1e-3

# ==========================================
# 2. CUSTOM METRICS
# ==========================================
def masked_loss(y_true, y_pred):
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction='none'
    )
    loss = loss_obj(y_true, y_pred)
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, dtype='int64')
    pred_id = tf.argmax(y_pred, axis=-1)
    correct = tf.math.equal(y_true, pred_id)
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    correct = tf.math.logical_and(correct, mask)
    correct = tf.cast(correct, dtype='float32')
    mask = tf.cast(mask, dtype='float32')
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)

# ==========================================
# 3. DATA PIPELINE
# ==========================================
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return img

def make_dataset(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    df["full_image_path"] = df["image_path"].apply(lambda x: os.path.join(IMG_DIR, str(x)))
    
    def parse_sequence(s):
        return [int(x) for x in re.findall(r'\d+', str(s))]
    
    df["caption_seq"] = df["caption_seq"].apply(parse_sequence)
    df = df[df["full_image_path"].apply(os.path.exists)].reset_index(drop=True)

    image_paths = df["full_image_path"].values
    captions = list(df['caption_seq'].values)
    captions = [c[:MAX_LEN] for c in captions] 

    cap_in = [c[:-1] for c in captions]  
    cap_out = [c[1:] for c in captions]  

    ds = tf.data.Dataset.from_tensor_slices((image_paths, cap_in, cap_out))

    def map_func(img_path, c_in, c_out):
        img = load_image(img_path)
        c_in = tf.pad(c_in, [[0, MAX_LEN - tf.shape(c_in)[0]]])
        c_out = tf.pad(c_out, [[0, MAX_LEN - tf.shape(c_out)[0]]])
        c_in = tf.ensure_shape(c_in, [MAX_LEN])
        c_out = tf.ensure_shape(c_out, [MAX_LEN])
        return (img, c_in), c_out

    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ==========================================
# 4. MODEL COMPONENTS
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
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
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
        return self.projection(patch) + self.position_embedding(positions)
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
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x, use_causal_mask=True)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, context)
    res = x + res
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer():
    image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image_input")
    caption_input = layers.Input(shape=(MAX_LEN,), name="caption_input")

    patches = Patches(PATCH_SIZE)(image_input)
    encoded_patches = PatchEncoder(NUM_PATCHES, EMBED_DIM)(patches)

    for _ in range(NUM_LAYERS):
        encoded_patches = transformer_encoder_block(
            encoded_patches, head_size=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT
        )

    caption_embed = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(caption_input)
    positions = tf.range(start=0, limit=MAX_LEN, delta=1)
    pos_embed = layers.Embedding(input_dim=MAX_LEN, output_dim=EMBED_DIM)(positions)
    caption_final = caption_embed + pos_embed

    for _ in range(NUM_LAYERS):
        caption_final = transformer_decoder_block(
            caption_final, context=encoded_patches, 
            head_size=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT
        )

    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(caption_final)

    model = keras.Model(inputs=[image_input, caption_input], outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    return model

# ==========================================
# 5. BEAM SEARCH GENERATION (THE FIX)
# ==========================================
def _load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

def beam_search_caption(model, tokenizer, image_path, beam_width=3, alpha=0.7):
    """
    Beam Search with Length Penalty (alpha).
    alpha: 0.6-0.9 encourages concise captions.
    """
    img = load_image(image_path)
    img = tf.expand_dims(img, axis=0)
    
    start_id = tokenizer.word_index.get("<start>")
    end_id = tokenizer.word_index.get("<end>")
    
    # Sequence: (log_prob, [tokens])
    sequences = [[0.0, [start_id]]]
    
    for _ in range(MAX_LEN):
        all_candidates = []
        
        for score, seq in sequences:
            # If sequence ended, keep it
            if seq[-1] == end_id:
                all_candidates.append([score, seq])
                continue
            
            # Predict next token
            cap_seq = tf.constant(seq)
            pad_len = MAX_LEN - tf.shape(cap_seq)[0]
            cap_in = tf.pad(cap_seq, [[0, pad_len]])
            cap_in = tf.expand_dims(cap_in, 0)
            
            preds = model.predict([img, cap_in], verbose=0)
            logits = preds[0, len(seq)-1, :]
            
            # Repetition Penalty (Soft)
            if len(seq) > 1: logits[seq[-1]] -= 10.0 # Discourage repeating immediate word
            
            # Get Top K candidates
            top_k_indices = np.argsort(logits)[-beam_width:]
            
            for idx in top_k_indices:
                # Add log probability (math.log for stability)
                prob = logits[idx]
                # Softmax approx (simplified for beam) or just use logits score
                candidate_score = score + prob 
                candidate_seq = seq + [idx]
                all_candidates.append([candidate_score, candidate_seq])
        
        # Select top K sequences
        ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        sequences = ordered[:beam_width]
        
        # Stop if all top sequences have ended
        if all(seq[1][-1] == end_id for seq in sequences):
            break
            
    # Select best sequence with Length Normalization
    # Score = log_prob / (length ^ alpha)
    best_score = -float('inf')
    best_seq = None
    
    for score, seq in sequences:
        # Length penalty calculation
        length_penalty = math.pow(len(seq), alpha)
        normalized_score = score / length_penalty
        
        if normalized_score > best_score:
            best_score = normalized_score
            best_seq = seq
            
    words = [tokenizer.index_word.get(i, "") for i in best_seq if i not in [start_id, end_id]]
    return " ".join(words)

def evaluate_test_set(model):
    df = pd.read_csv(TEST_CSV).head(10)
    df["full_path"] = df["image_path"].apply(lambda x: os.path.join(IMG_DIR, str(x)))
    df = df[df["full_path"].apply(os.path.exists)]
    
    tokenizer = _load_tokenizer()
    results_path = os.path.join(RESULTS_DIR, "final_transformer_examples.txt")
    
    with open(results_path, "w") as f:
        f.write("FINAL EVALUATION (BEAM SEARCH WIDTH=3)\n========================================\n")
        print("\n--- Generating Examples with Beam Search ---")
        for _, row in df.iterrows():
            # Use Beam Search instead of greedy
            pred = beam_search_caption(model, tokenizer, row["full_path"], beam_width=3)
            log = f"Style: {row['art_style']}\nRef: {row['utterance']}\nPred: {pred}\n{'-'*40}\n"
            print(log)
            f.write(log)
    print(f"Saved examples to {results_path}")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- 1. Training Smaller Model ---")
    BATCH_SIZE = 32
    train_ds = make_dataset(TRAIN_CSV, BATCH_SIZE)
    val_ds = make_dataset(VAL_CSV, BATCH_SIZE)

    model = build_transformer()
    
    # Train for 10 epochs
    model.fit(
        train_ds, validation_data=val_ds, epochs=10,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    )
    
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("--- 2. Evaluating ---")
    evaluate_test_set(model)