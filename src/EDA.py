# EDA 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from collections import Counter
import nltk
from nltk.util import ngrams

# Download NLTK data (handles 'lookup' errors if not present)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --- CONFIGURATION ---
DATA_DIR = os.path.join("..", "Data")
INPUT_CSV = os.path.join(DATA_DIR, "initial_dataset.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "artemis_10k_sampled.csv")
# Logic: dropping.py creates 'Initial_Artworks_folder' as a sibling to 'Data'
IMG_DIR = os.path.join("..", "Initial_Artworks_folder") 
VIZ_DIR = os.path.join("..", "Visualisations")
TARGET_SIZE = 10000
RANDOM_SEED = 42

os.makedirs(VIZ_DIR, exist_ok=True)

# ==========================================
# PART 1: STRATIFIED SAMPLING
# ==========================================
print(f"--- PART 1: LOADING & SAMPLING ---")
if not os.path.exists(INPUT_CSV):
    print(f"Error: {INPUT_CSV} not found. Run dropping.py first.")
    exit()

df = pd.read_csv(INPUT_CSV)
style_counts = df['art_style'].value_counts()

# Calculate 'Cap' to hit ~10k target
def find_optimal_cutoff(counts, target):
    low, high = 1, counts.max()
    best_cutoff = high
    closest_diff = float('inf')
    
    while low <= high:
        mid = (low + high) // 2
        current_sum = sum([min(c, mid) for c in counts])
        if abs(target - current_sum) < closest_diff:
            closest_diff = abs(target - current_sum)
            best_cutoff = mid
        if current_sum < target: low = mid + 1
        else: high = mid - 1
    return best_cutoff

cutoff = find_optimal_cutoff(style_counts, TARGET_SIZE)
print(f"Sampling Cap Calculated: {cutoff} images per style.")

# Perform Sampling
sampled_dfs = []
for style, count in style_counts.items():
    style_subset = df[df['art_style'] == style]
    if count <= cutoff:
        sampled_dfs.append(style_subset)
    else:
        sampled_dfs.append(style_subset.sample(n=cutoff, random_state=RANDOM_SEED))

final_df = pd.concat(sampled_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(final_df)} samples to {OUTPUT_CSV}")

# Save Distribution Plot
plt.figure(figsize=(14, 8))
sns.barplot(x=final_df['art_style'].value_counts().index, y=final_df['art_style'].value_counts().values, palette='magma')
plt.xticks(rotation=90)
plt.title(f'Final Sampled Distribution (Total: {len(final_df)})')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '02_distribution_sampled.png'))
plt.close()

# ==========================================
# PART 2: TEXT ANALYSIS (Captions)
# ==========================================
print(f"\n--- PART 2: TEXT ANALYSIS ---")

# 1. Caption Length & Vocab 
final_df['caption_len'] = final_df['utterance'].astype(str).apply(lambda x: len(x.split()))
avg_len = final_df['caption_len'].mean()
max_len = final_df['caption_len'].max()
vocab = set(" ".join(final_df['utterance'].astype(str)).lower().split())

print(f"Average Caption Length: {avg_len:.2f} words")
print(f"Max Caption Length: {max_len} words")
print(f"Vocabulary Size: {len(vocab)} unique words")

# Plot Caption Length
plt.figure(figsize=(10, 5))
sns.histplot(final_df['caption_len'], bins=30, kde=True, color='teal')
plt.title('Distribution of Caption Lengths (Sampled Dataset)')
plt.axvline(avg_len, color='r', linestyle='--', label=f'Avg: {avg_len:.1f}')
plt.legend()
plt.savefig(os.path.join(VIZ_DIR, '03_caption_lengths.png'))
plt.close()

# 2. Common Words & Bigrams 
all_tokens = " ".join(final_df['utterance'].astype(str)).lower().split()
# Remove stopwords for cleaner analysis
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in all_tokens if w.isalnum() and w not in stop_words]

# Top 10 Words
common_words = Counter(filtered_tokens).most_common(10)
print(f"Top 10 Words: {common_words}")

# Top 10 Bigrams
common_bigrams = Counter(ngrams(filtered_tokens, 2)).most_common(10)
print(f"Top 10 Bigrams: {common_bigrams}")

# Plot Bigrams
bigram_labels = [f"{w1} {w2}" for (w1, w2), freq in common_bigrams]
bigram_freqs = [freq for (w1, w2), freq in common_bigrams]

plt.figure(figsize=(12, 6))
sns.barplot(x=bigram_freqs, y=bigram_labels, palette='viridis')
plt.title('Top 10 Common Bigrams')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '04_top_bigrams.png'))
plt.close()

# ==========================================
# PART 3: DIVERSITY & PATTERNS
# ==========================================
print(f"\n--- PART 3: DIVERSITY & PATTERNS ---")

# 1. Caption Diversity 
# Ratio of Unique Captions to Total Captions per Style
diversity = final_df.groupby('art_style')['utterance'].nunique() / final_df.groupby('art_style')['utterance'].count()
diversity = diversity.sort_values()

plt.figure(figsize=(12, 6))
diversity.plot(kind='barh', color='salmon')
plt.title('Caption Diversity Score by Art Style (Unique/Total)')
plt.xlabel('Diversity Ratio (Higher is better)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '05_caption_diversity.png'))
plt.close()

# 2. Emotion vs Style Heatmap 
# Shows if certain styles have specific emotional biases
heatmap_data = pd.crosstab(final_df['art_style'], final_df['emotion'], normalize='index')

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Emotion Distribution across Art Styles (Normalized)')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '06_emotion_style_heatmap.png'))
plt.close()

# ==========================================
# PART 4: IMAGE VISUALIZATION
# ==========================================
print(f"\n--- PART 4: VISUALIZING SAMPLES ---")

def visualize_grid(df, img_root, num_samples=5):
    samples = df.sample(num_samples, random_state=RANDOM_SEED)
    plt.figure(figsize=(20, 10)) # Wide figure
    
    for idx, (i, row) in enumerate(samples.iterrows()):
        # Construct path: Root / Style / Painting.jpg
        img_path = os.path.join(img_root, row['art_style'], row['painting'] + '.jpg')
        
        # Fallback for extensions
        if not os.path.exists(img_path):
            # Try finding file ignoring extension case
            style_dir = os.path.join(img_root, row['art_style'])
            if os.path.exists(style_dir):
                for f in os.listdir(style_dir):
                    if f.startswith(row['painting']):
                        img_path = os.path.join(style_dir, f)
                        break
        
        try:
            with Image.open(img_path) as img:
                plt.subplot(1, num_samples, idx + 1)
                plt.imshow(img)
                plt.axis('off')
                # Wrap text for cleaner title
                title_text = f"[{row['emotion']}]\n{row['utterance'][:60]}..."
                plt.title(title_text, fontsize=10)
        except Exception as e:
            print(f"Could not load {img_path}: {e}")
            
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, '07_sample_visualizations.png'))
    plt.close()
    print(f"Saved sample visualization to {VIZ_DIR}")

# Check if image directory exists before running viz
if os.path.exists(IMG_DIR):
    visualize_grid(final_df, IMG_DIR, num_samples=5)
else:
    print(f"WARNING: Image directory {IMG_DIR} not found. Skipping visualization.")

print(f"\nSUCCESS: EDA Completed. Check '{VIZ_DIR}' for all plots.")