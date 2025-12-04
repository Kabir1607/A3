import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Set

import pandas as pd


# Dropping artworks that are not both in ArtEmis and Wikiart dataset: 
# As in, if an artwork is not both in the csv file and a correspondong Artwork does not exist, if both are not there, then either drop that row in the csv, or delete that file in the Artworks folder: 
# Essenstially, make sure that there is a 1-1 mapping between the artworks in the csv file (dont count repetitions of the same artwork) and the artworks in the Artworks folder.
# save the new csv as: "initial_dataset.csv" in the Data folder and the new Artworks folder as: "Initial_Artworks_folder": 


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "..", "Data")
ORIG_CSV = os.path.join(DATA_DIR, "artemis_dataset_release_v0.csv")

ARTWORKS_ROOT = os.path.join(THIS_DIR, "..", "Artworks")
OUTPUT_CSV = os.path.join(DATA_DIR, "initial_dataset.csv")
OUTPUT_ARTWORKS_ROOT = os.path.join(THIS_DIR, "..", "Initial_Artworks_folder")
NOTES_PATH = os.path.join(THIS_DIR, "..", "Visualisations", "Notes.txt")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_artemis_paintings(csv_path: str = ORIG_CSV) -> Set[str]:
    """Return the set of unique painting IDs present in the ArtEmis CSV."""
    df = pd.read_csv(
        csv_path,
        usecols=["painting"],
        dtype={"painting": "string"},
        low_memory=False,
    )
    return set(df["painting"].dropna().unique())


def _collect_wikiart_files(root_dir: str = ARTWORKS_ROOT) -> Dict[str, str]:
    """
    Walk the Artworks folder and build a mapping from painting id (filename without extension)
    to its full path on disk.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Artworks root directory not found: {root_dir}")

    mapping: Dict[str, str] = {}
    for style_name in os.listdir(root_dir):
        style_path = os.path.join(root_dir, style_name)
        if not os.path.isdir(style_path):
            continue

        for dirpath, _, filenames in os.walk(style_path):
            for fname in filenames:
                base, ext = os.path.splitext(fname)
                if ext.lower() not in IMAGE_EXTS:
                    continue
                # if duplicates exist, keep the first one we encounter
                mapping.setdefault(base, os.path.join(dirpath, fname))

    return mapping


def build_initial_dataset(
    csv_in: str = ORIG_CSV,
    csv_out: str = OUTPUT_CSV,
    artworks_root: str = ARTWORKS_ROOT,
    artworks_out_root: str = OUTPUT_ARTWORKS_ROOT,
) -> None:
    """
    Create a filtered CSV and artwork folder such that there is a 1-1 mapping between
    paintings in the CSV and image files in the Artworks folder (ignoring repetitions
    of the same painting in the CSV).
    """
    # Load CSV and collect unique painting ids
    df = pd.read_csv(
        csv_in,
        dtype={
            "art_style": "category",
            "painting": "string",
            "emotion": "category",
            "utterance": "string",
            "repetition": "int16",
        },
        low_memory=False,
    )
    original_rows = len(df)
    artemis_paintings = set(df["painting"].dropna().unique())

    # Map available image files (without extension) to their paths
    wikiart_mapping = _collect_wikiart_files(artworks_root)
    wikiart_paintings = set(wikiart_mapping.keys())

    # Keep only paintings that exist in BOTH datasets
    valid_paintings = artemis_paintings & wikiart_paintings
    deleted_artworks = len(wikiart_paintings - valid_paintings)

    # Filter CSV rows to keep only valid paintings
    filtered_df = df[df["painting"].isin(valid_paintings)].copy()
    kept_rows = len(filtered_df)
    rows_dropped = original_rows - kept_rows

    # Ensure output directory for CSV exists
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    filtered_df.to_csv(csv_out, index=False)

    # Prepare new artworks folder
    if os.path.exists(artworks_out_root):
        shutil.rmtree(artworks_out_root)
    os.makedirs(artworks_out_root, exist_ok=True)

    # Pre-create destination style folders once
    style_dirs: Dict[str, str] = {}
    painting_to_style: Dict[str, str] = {}
    for painting_id in valid_paintings:
        src_path = wikiart_mapping[painting_id]
        rel_from_root = os.path.relpath(src_path, artworks_root)
        style_name = rel_from_root.split(os.sep)[0]
        painting_to_style[painting_id] = style_name
        if style_name not in style_dirs:
            dst_style_dir = os.path.join(artworks_out_root, style_name)
            os.makedirs(dst_style_dir, exist_ok=True)
            style_dirs[style_name] = dst_style_dir

    # Copy artworks in parallel to leverage I/O concurrency
    def _copy_artwork(painting_id: str) -> None:
        src_path = wikiart_mapping[painting_id]
        style_name = painting_to_style[painting_id]
        dst_path = os.path.join(
            style_dirs[style_name], os.path.basename(src_path)
        )
        shutil.copy2(src_path, dst_path)

    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(_copy_artwork, valid_paintings)

    # Console summary
    print(f"Filtered CSV written to: {csv_out}")
    print(f"Filtered artworks copied to: {artworks_out_root}")
    print(f"Original CSV rows: {original_rows}")
    print(f"Kept CSV rows: {kept_rows}")
    print(f"Rows dropped from CSV: {rows_dropped}")
    print(f"Number of unique paintings kept: {len(valid_paintings)}")
    print(f"Estimated number of artworks deleted: {deleted_artworks}")

    # Write / append summary to Visualisations/Notes.txt
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    with open(NOTES_PATH, "a", encoding="utf-8") as notes_file:
        notes_file.write(
            "Initial dataset creation summary\n"
            f"- Original CSV rows: {original_rows}\n"
            f"- Kept CSV rows: {kept_rows}\n"
            f"- Rows dropped from CSV: {rows_dropped}\n"
            f"- Unique paintings kept: {len(valid_paintings)}\n"
            f"- Estimated artworks deleted from WikiArt folder: {deleted_artworks}\n\n"
        )


if __name__ == "__main__":
    build_initial_dataset()
