import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import numpy as np
import pandas as pd
import os
import ast
import re
import pickle

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = os.path.join("..", "Data")
IMG_DIR = os.path.join("..", "Initial_Artworks_folder")
RESULTS_DIR = os.path.join("..", "Results")
TRAIN_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_split.csv")
TEST_CSV = os.path.join(DATA_DIR, "test_split.csv")
FULL_SAMPLE_CSV = os.path.join(DATA_DIR, "artemis_10k_sampled.csv")

# Create Results Directory
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_SIZE = 224
PATCH_SIZE = 32
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
VOCAB_SIZE = 8001
MAX_LEN = 40

# Build a consistent emotion vocabulary across splits
if os.path.exists(FULL_SAMPLE_CSV):
    _emotion_df = pd.read_csv(FULL_SAMPLE_CSV, usecols=["emotion"])
    EMOTIONS = sorted(_emotion_df["emotion"].dropna().unique().tolist())
else:
    # Fallback: will be populated lazily from the train CSV if needed
    EMOTIONS = []

EMOTION_TO_ID = {e: idx for idx, e in enumerate(EMOTIONS)}
NUM_EMOTIONS = len(EMOTIONS)

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.pkl")

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

    # Build / update emotion vocabulary lazily if not loaded from full sample CSV
    global EMOTIONS, EMOTION_TO_ID, NUM_EMOTIONS
    if not EMOTIONS:
        emotions = sorted(df["emotion"].dropna().unique().tolist())
        EMOTIONS = emotions
        EMOTION_TO_ID = {e: idx for idx, e in enumerate(EMOTIONS)}
        NUM_EMOTIONS = len(EMOTIONS)

    # Drop rows whose image file does not exist (robust to missing/deleted artworks)
    df["full_image_path"] = df["image_path"].apply(
        lambda x: os.path.join(IMG_DIR, str(x))
    )
    df = df[df["full_image_path"].apply(os.path.exists)].reset_index(drop=True)

    # Map emotions to integer IDs
    df["emotion_id"] = df["emotion"].map(EMOTION_TO_ID).astype("int32")

    # Robust Parser: Extracts numbers even if format is messy
    def parse_sequence(s):
        return [int(x) for x in re.findall(r'\d+', str(s))]
    
    df["caption_seq"] = df["caption_seq"].apply(parse_sequence)

    image_paths = df["full_image_path"].values
    emotion_ids = df["emotion_id"].values
    captions = list(df['caption_seq'].values)

    # --- SAFETY FIX: TRUNCATE SEQUENCES ---
    # Ensure no sequence exceeds MAX_LEN before processing
    # This prevents the "Negative Padding" crash
    captions = [c[:MAX_LEN] for c in captions] 
    # --------------------------------------

    cap_in = [c[:-1] for c in captions]
    cap_out = [c[1:] for c in captions]

    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, emotion_ids, cap_in, cap_out)
    )

    def map_func(img_path, emotion_id, c_in, c_out):
        img = load_image(img_path)
        # Dynamic padding calculation
        padding_in = MAX_LEN - tf.shape(c_in)[0]
        padding_out = MAX_LEN - tf.shape(c_out)[0]
        
        c_in = tf.pad(c_in, [[0, padding_in]])
        c_out = tf.pad(c_out, [[0, padding_out]])
        
        c_in = tf.ensure_shape(c_in, [MAX_LEN])
        c_out = tf.ensure_shape(c_out, [MAX_LEN])
        return (img, emotion_id, c_in), c_out

    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ==========================================
# 3b. EVALUATION HELPERS (BLEU / ROUGE)
# ==========================================

def _load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Run Preprocessing.py first.")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def _tokens_to_caption(token_ids, tokenizer):
    # Map ids back to words, stopping at <end>
    id_to_word = {v: k for k, v in tokenizer.word_index.items()}
    words = []
    for tid in token_ids:
        if tid == 0:
            continue
        word = id_to_word.get(int(tid), "")
        if word == "<start>":
            continue
        if word == "<end>":
            break
        if word:
            words.append(word)
    return " ".join(words)


def generate_caption(model, tokenizer, image_path, emotion_str, max_len=40):
    """Greedy decoding: given image + emotion string -> caption text."""
    if emotion_str not in EMOTION_TO_ID:
        raise ValueError(f"Unknown emotion '{emotion_str}'. Known: {list(EMOTION_TO_ID.keys())}")
    emotion_id = EMOTION_TO_ID[emotion_str]

    # Prepare image
    img = load_image(image_path)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)

    # Start token
    tokenizer = tokenizer
    start_id = tokenizer.word_index.get("<start>")
    end_id = tokenizer.word_index.get("<end>")
    if start_id is None or end_id is None:
        raise ValueError("Tokenizer must contain <start> and <end> tokens.")

    seq = [start_id]
    for i in range(1, max_len):
        cap_in = tf.constant(seq, dtype=tf.int32)
        cap_in = tf.pad(cap_in, [[0, max_len - tf.shape(cap_in)[0]]])
        cap_in = tf.expand_dims(cap_in, axis=0)  # (1, max_len)

        emotion_tensor = tf.constant([emotion_id], dtype=tf.int32)

        preds = model.predict([img, emotion_tensor, cap_in], verbose=0)
        next_id = int(tf.argmax(preds[0, i - 1, :]))
        seq.append(next_id)
        if next_id == end_id:
            break

    return _tokens_to_caption(seq, tokenizer)


def _rouge1_f1(reference_tokens, predicted_tokens):
    """Simple ROUGE-1 F1 (unigram overlap)."""
    ref_counts = {}
    for w in reference_tokens:
        ref_counts[w] = ref_counts.get(w, 0) + 1
    pred_counts = {}
    for w in predicted_tokens:
        pred_counts[w] = pred_counts.get(w, 0) + 1

    overlap = 0
    for w, c in ref_counts.items():
        if w in pred_counts:
            overlap += min(c, pred_counts[w])

    if overlap == 0:
        return 0.0

    precision = overlap / max(1, len(predicted_tokens))
    recall = overlap / max(1, len(reference_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_on_split(model, csv_path, num_samples=500):
    """Compute BLEU and ROUGE-1 on a CSV split (e.g., test_split)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter to rows with existing images and known emotions
    df["full_image_path"] = df["image_path"].apply(
        lambda x: os.path.join(IMG_DIR, str(x))
    )
    df = df[df["full_image_path"].apply(os.path.exists)]
    df = df[df["emotion"].isin(EMOTIONS)].reset_index(drop=True)

    if num_samples is not None:
        df = df.head(num_samples)

    tokenizer = _load_tokenizer()
    smoothie = SmoothingFunction().method1

    bleu1_scores, bleu2_scores, bleu4_scores, rouge1_scores = [], [], [], []
    examples = []

    for _, row in df.iterrows():
        ref_caption = str(row["utterance"]).lower()
        ref_tokens = ref_caption.split()

        pred_caption = generate_caption(
            model, tokenizer, row["full_image_path"], row["emotion"], max_len=MAX_LEN
        ).lower()
        pred_tokens = pred_caption.split()

        if not pred_tokens or not ref_tokens:
            continue

        bleu1 = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie,
        )
        bleu2 = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothie,
        )
        bleu4 = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
        rouge1 = _rouge1_f1(ref_tokens, pred_tokens)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)
        rouge1_scores.append(rouge1)

        if len(examples) < 5:
            examples.append(
                {
                    "emotion": row["emotion"],
                    "reference": ref_caption,
                    "predicted": pred_caption,
                }
            )

    results = {
        "num_samples": len(bleu1_scores),
        "bleu1": float(np.mean(bleu1_scores)) if bleu1_scores else 0.0,
        "bleu2": float(np.mean(bleu2_scores)) if bleu2_scores else 0.0,
        "bleu4": float(np.mean(bleu4_scores)) if bleu4_scores else 0.0,
        "rouge1_f1": float(np.mean(rouge1_scores)) if rouge1_scores else 0.0,
        "examples": examples,
    }
    return results

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
    image_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image")
    emotion_input = layers.Input(shape=(), dtype="int32", name="emotion")
    caption_input = layers.Input(shape=(MAX_LEN,), name="caption")

    # Image Encoding
    patches = Patches(PATCH_SIZE)(image_input)
    encoded_patches = PatchEncoder(NUM_PATCHES, embed_dim)(patches)

    for _ in range(num_layers):
        encoded_patches = transformer_encoder_block(
            encoded_patches, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate
        )

    # Text + Emotion Encoding
    caption_embed = layers.Embedding(VOCAB_SIZE, embed_dim)(caption_input)
    positions = tf.range(start=0, limit=MAX_LEN, delta=1)
    pos_embed = layers.Embedding(input_dim=MAX_LEN, output_dim=embed_dim)(positions)

    # Emotion embedding: condition the caption generation on emotion
    if NUM_EMOTIONS > 0:
        emotion_embed_layer = layers.Embedding(NUM_EMOTIONS, embed_dim)
        emotion_vec = emotion_embed_layer(emotion_input)           # (batch, embed_dim)
        # Repeat across time steps using Keras layer ops (no raw tf.* on KerasTensor)
        emotion_vec = layers.RepeatVector(MAX_LEN)(emotion_vec)    # (batch, MAX_LEN, embed_dim)
        caption_final = caption_embed + pos_embed + emotion_vec
    else:
        caption_final = caption_embed + pos_embed

    for _ in range(num_layers):
        caption_final = transformer_decoder_block(
            caption_final, context=encoded_patches, 
            head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout_rate
        )

    outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(caption_final)

    model = keras.Model(
        inputs=[image_input, emotion_input, caption_input],
        outputs=outputs,
    )
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

    # Collect ALL trials (including failed ones) for a complete log
    all_trials = list(tuner.oracle.trials.values())

    with open(results_path, "w") as f:
        f.write("TRANSFORMER HYPERPARAMETER TUNING RESULTS\n")
        f.write("=========================================\n\n")
        for i, trial in enumerate(all_trials):
            f.write(f"Trial #{i+1}\n")
            f.write(f"  Trial ID: {trial.trial_id}\n")
            f.write(f"  Status  : {trial.status}\n")
            f.write(f"  Score   : {trial.score}\n")
            f.write("  Hyperparameters:\n")
            for hp, value in trial.hyperparameters.values.items():
                f.write(f"    - {hp}: {value}\n")
            f.write("-" * 40 + "\n")

    print(f"Detailed results saved to {results_path}")

    # Save best model only if at least one successful trial exists
    successful_trials = [t for t in all_trials if t.score is not None]
    if successful_trials:
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        best_model.save(os.path.join(DATA_DIR, "best_transformer_model.keras"))
        print("Best model saved to ../Data/best_transformer_model.keras")

        # ===============================
        # Evaluate best model on test set
        # ===============================
        if os.path.exists(TEST_CSV):
            print("\n--- Evaluating best Transformer on test_split.csv ---")
            eval_results = evaluate_on_split(
                best_model, TEST_CSV, num_samples=500
            )
            eval_path = os.path.join(
                RESULTS_DIR, "transformer_evaluation_results.txt"
            )
            with open(eval_path, "w", encoding="utf-8") as f:
                f.write("TRANSFORMER EVALUATION RESULTS (TEST SPLIT)\n")
                f.write("===========================================\n\n")
                f.write(f"Num samples evaluated: {eval_results['num_samples']}\n")
                f.write(f"BLEU-1: {eval_results['bleu1']:.4f}\n")
                f.write(f"BLEU-2: {eval_results['bleu2']:.4f}\n")
                f.write(f"BLEU-4: {eval_results['bleu4']:.4f}\n")
                f.write(f"ROUGE-1 F1: {eval_results['rouge1_f1']:.4f}\n\n")
                f.write("Sample qualitative examples:\n")
                for ex in eval_results["examples"]:
                    f.write(f"- Emotion   : {ex['emotion']}\n")
                    f.write(f"  Reference : {ex['reference']}\n")
                    f.write(f"  Predicted : {ex['predicted']}\n")
                    f.write("-" * 40 + "\n")

            print(f"Evaluation results saved to {eval_path}")
        else:
            print(f"Test CSV not found at {TEST_CSV}, skipping evaluation.")
    else:
        print("No successful trials to save a best model from.")