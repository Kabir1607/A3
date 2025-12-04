import os
from collections import defaultdict

import pandas as pd


# Looking and exploring the dataset to see if all is in order: 

# Counting Number of Artworks per Style in ArtEmis (the csv file):
# Ignore repetitions, just count the number of unique artworks per style: 

# Counting Number of Artworks per Style in wikiart (the folders in Artworks folder): 


# Resolve paths relative to this file so the script works no matter
# where it is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CSV = os.path.join(_THIS_DIR, "..", "Data", "artemis_dataset_release_v0.csv")
_ARTWORKS_ROOT = os.path.join(_THIS_DIR, "..", "Artworks")


def count_unique_artworks_per_style_artemis(csv_path: str = _DATA_CSV) -> pd.Series:
    """
    Counting Number of Artworks per Style in ArtEmis (the csv file).
    Ignore repetitions, just count the number of the *unique* artworks per style.
    """
    df = pd.read_csv(csv_path)

    # We assume: one row per (art_style, painting, emotion, utterance, repetition).
    # Unique artworks per style = number of distinct painting IDs for each art_style.
    counts = (
        df.groupby("art_style")["painting"]
        .nunique()
        .sort_values(ascending=False)
    )
    return counts


def count_artworks_per_style_wikiart(root_dir: str = _ARTWORKS_ROOT) -> dict:
    """
    Counting Number of Artworks per Style in wikiart (the folders in Artworks folder).

    Each immediate sub-folder of `root_dir` is treated as an art style, and we count
    the number of image files contained within that sub-folder (recursively).
    """
    # Common image extensions in the dataset
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    style_counts: dict[str, int] = defaultdict(int)

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Artworks root directory not found: {root_dir}")

    for style_name in os.listdir(root_dir):
        style_path = os.path.join(root_dir, style_name)
        if not os.path.isdir(style_path):
            continue

        count = 0
        for dirpath, _, filenames in os.walk(style_path):
            for fname in filenames:
                _, ext = os.path.splitext(fname)
                if ext.lower() in image_exts:
                    count += 1

        style_counts[style_name] = count

    return dict(style_counts)


if __name__ == "__main__":
    # Counting Number of Artworks per Style in ArtEmis (the csv file)
    artemis_counts = count_unique_artworks_per_style_artemis()
    print("Unique artworks per style in ArtEmis (CSV):")
    print(artemis_counts)
    print()

    # Counting Number of Artworks per Style in wikiart (the folders in Artworks folder)
    wikiart_counts = count_artworks_per_style_wikiart()
    print("Artworks per style in WikiArt (folder structure):")
    for style, count in sorted(wikiart_counts.items()):
        print(f"{style}: {count}")

