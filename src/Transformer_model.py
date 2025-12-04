import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
import pandas as pd
import os
import ast
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = os.path.join("..", "Data")
IMG_DIR = os.path.join("..", "Initial_Artworks_folder")
RESULTS_DIR = os.path.join("..", "Results")
TRAIN_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_split.csv")

# Create Results Directory
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 224
PATCH_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
VOCAB_SIZE = 8001
MAX_LEN = 40

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    return img

def make_dataset(csv_path, batch_size):
    df = pd.read_csv(csv_path)
    
    # Robust Parser: Extracts numbers even if format is messy
    def parse_sequence(s):
        return [int(x) for x in re.findall(r'\d+', str(s))]
    
    df['caption_seq'] = df['caption_seq'].apply(parse_sequence)
    
    image_paths = df['image_path'].apply(lambda x: os.path.join(IMG_DIR, x)).values
    captions = list(df['caption_seq'].values)

    # --- SAFETY FIX: TRUNCATE SEQUENCES ---
    # Ensure no sequence exceeds MAX_LEN before processing
    # This prevents the "Negative Padding" crash
    captions = [c[:MAX_LEN] for c in captions] 
    # --------------------------------------

    cap_in = [c[:-1] for c in captions]
    cap_out = [c[1:] for c in captions]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, cap_in, cap_out))

    def map_func(img_path, c_in, c_out):
        img = load_image(img_path)
        # Dynamic padding calculation
        padding_in = MAX_LEN - tf.shape(c_in)[0]
        padding_out = MAX_LEN - tf.shape(c_out)[0]
        
        c_in = tf.pad(c_in, [[0, padding_in]])
        c_out = tf.pad(c_out, [[0, padding_out]])
        
        c_in = tf.ensure_shape(c_in, [MAX_LEN])
        c_out = tf.ensure_shape(c_out, [MAX_LEN])
        return (img, c_in), c_out

    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ==========================================
# 3. MODEL BLOCKS
# ==========================================
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

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
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, context)
    res = x + res
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# ==========================================
# 4. BUILD MODEL (Keras Tuner)
# ==========================================
def build_model(hp):
    # Tunable Params
    embed_dim = hp.Int('embed_dim', min_value=64, max_value=256, step=64)
    num_heads = hp.Choice('num_heads', values=[2, 4, 8])
    ff_dim = hp.Int('ff_dim', min_value=64, max_value=256, step=64)
    num_layers = hp.Int('num_layers', min_value=2, max_value=4, step=2)
    dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1)
    learning_rate = hp.Choice('lr', values=[1e-3, 5e-4])

    # Inputs
    image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    caption_input = layers.Input(shape=(MAX_LEN,))

    # Image Encoding
    patches = Patches(PATCH_SIZE)(image_input)
    encoded_patches = PatchEncoder(NUM_PATCHES, embed_dim)(patches)

    for _ in range(num_layers):
        encoded_patches = transformer_encoder_block(
            encoded_patches, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate
        )

    # Text Encoding
    caption_embed = layers.Embedding(VOCAB_SIZE, embed_dim)(caption_input)
    positions = tf.range(start=0, limit=MAX_LEN, delta=1)
    pos_embed = layers.Embedding(input_dim=MAX_LEN, output_dim=embed_dim)(positions)
    caption_final = caption_embed + pos_embed

    for _ in range(num_layers):
        caption_final = transformer_decoder_block(
            caption_final, context=encoded_patches, 
            head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate
        )

    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(caption_final)

    model = keras.Model(inputs=[image_input, caption_input], outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ==========================================
# 5. EXECUTION & LOGGING
# ==========================================
if __name__ == "__main__":
    print("--- Starting Keras Tuner ---")
    
    BATCH_SIZE = 32
    train_ds = make_dataset(TRAIN_CSV, BATCH_SIZE)
    val_ds = make_dataset(VAL_CSV, BATCH_SIZE)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=3,           
        executions_per_trial=1, 
        directory='tuner_dir',  
        project_name='artemis_transformer_tuning_v3'
    )

    tuner.search_space_summary()

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=3, 
        callbacks=[keras.callbacks.EarlyStopping(patience=1)]
    )

    print("\n--- Saving Results ---")
    results_path = os.path.join(RESULTS_DIR, "transformer_tuning_results.txt")
    
    all_trials = tuner.oracle.get_best_trials(num_trials=10)
    
    with open(results_path, "w") as f:
        f.write("TRANSFORMER HYPERPARAMETER TUNING RESULTS\n")
        f.write("=========================================\n\n")
        for i, trial in enumerate(all_trials):
            f.write(f"Rank: {i+1}\n")
            f.write(f"Trial ID: {trial.trial_id}\n")
            f.write(f"Validation Accuracy: {trial.score:.4f}\n")
            f.write("Hyperparameters:\n")
            for hp, value in trial.hyperparameters.values.items():
                f.write(f"  - {hp}: {value}\n")
            f.write("-" * 30 + "\n")
            
    print(f"Detailed results saved to {results_path}")

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    best_model.save(os.path.join(DATA_DIR, "best_transformer_model.keras"))
    print("Best model saved to ../Data/best_transformer_model.keras")