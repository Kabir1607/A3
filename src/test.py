import os

import pandas as pd
import tensorflow as tf

from Transformer_model import (
    DATA_DIR,
    IMG_DIR,
    TEST_CSV,
    EMOTIONS,
    MAX_LEN,
    Patches,
    PatchEncoder,
    transformer_encoder_block,
    transformer_decoder_block,
    generate_caption,
    _load_tokenizer,
)


def load_best_transformer_model():
    """Load the best saved transformer model with required custom objects."""
    model_path = os.path.join(DATA_DIR, "best_transformer_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Best transformer model not found at {model_path}. "
            "Run Transformer_model.py first to train and save the model."
        )

    custom_objects = {
        "Patches": Patches,
        "PatchEncoder": PatchEncoder,
        "transformer_encoder_block": transformer_encoder_block,
        "transformer_decoder_block": transformer_decoder_block,
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


def sample_three_examples():
    """Sample 3 random (image, emotion, caption) rows from the test split."""
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(
            f"Test CSV not found at {TEST_CSV}. Run Preprocessing.py first."
        )

    df = pd.read_csv(TEST_CSV)

    # Build full image path and filter to existing files and known emotions
    df["full_image_path"] = df["image_path"].apply(
        lambda x: os.path.join(IMG_DIR, str(x))
    )
    df = df[df["full_image_path"].apply(os.path.exists)]
    df = df[df["emotion"].isin(EMOTIONS)]

    if len(df) == 0:
        raise RuntimeError(
            "No valid samples found in test_split.csv with existing images and known emotions."
        )

    # If there are fewer than 3 valid samples, just use all of them
    n_samples = min(3, len(df))
    return df.sample(n=n_samples, random_state=42).reset_index(drop=True)


def main():
    print("--- Loading tokenizer and best transformer model ---")
    tokenizer = _load_tokenizer()
    model = load_best_transformer_model()

    print("--- Sampling 3 random test examples ---")
    samples = sample_three_examples()

    print("\n=== SAMPLE CAPTION GENERATION (IMAGE + EMOTION → CAPTION) ===\n")
    for idx, row in samples.iterrows():
        image_path = row["full_image_path"]
        emotion = row["emotion"]
        reference_caption = str(row["utterance"])

        generated_caption = generate_caption(
            model, tokenizer, image_path, emotion, max_len=MAX_LEN
        )

        print(f"Example #{idx + 1}")
        print(f"Image path : {image_path}")
        print(f"Emotion    : {emotion}")
        print(f"Reference  : {reference_caption}")
        print(f"Predicted  : {generated_caption}")
        print("-" * 60)


if __name__ == "__main__":
    main()

