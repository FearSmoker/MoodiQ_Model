#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# 12 Refined Mood Profiles (from model_service.py)
MOOD_PROFILES = {
    "Happy": {
        "danceability": (0.7, 0.9), "energy": (0.6, 0.85), "loudness": (0.7, 0.9),
        "speechiness": (0.05, 0.3), "acousticness": (0.1, 0.4), "instrumentalness": (0.0, 0.2),
        "liveness": (0.1, 0.4), "valence": (0.7, 1.0), "tempo": (0.6, 0.8), "spec_rate": (0.6, 0.8)
    },
    "Sad": {
        "danceability": (0.2, 0.5), "energy": (0.2, 0.45), "loudness": (0.2, 0.4),
        "speechiness": (0.02, 0.25), "acousticness": (0.4, 0.8), "instrumentalness": (0.1, 0.6),
        "liveness": (0.1, 0.3), "valence": (0.0, 0.3), "tempo": (0.3, 0.5), "spec_rate": (0.3, 0.5)
    },
    "Energetic": {
        "danceability": (0.6, 0.8), "energy": (0.8, 1.0), "loudness": (0.8, 1.0),
        "speechiness": (0.05, 0.25), "acousticness": (0.05, 0.3), "instrumentalness": (0.0, 0.1),
        "liveness": (0.2, 0.5), "valence": (0.6, 0.9), "tempo": (0.8, 1.0), "spec_rate": (0.7, 1.0)
    },
    "Calm": {
        "danceability": (0.3, 0.6), "energy": (0.2, 0.4), "loudness": (0.2, 0.5),
        "speechiness": (0.03, 0.15), "acousticness": (0.6, 0.9), "instrumentalness": (0.2, 0.6),
        "liveness": (0.1, 0.3), "valence": (0.4, 0.7), "tempo": (0.3, 0.5), "spec_rate": (0.3, 0.6)
    },
    "Focused": {
        "danceability": (0.4, 0.7), "energy": (0.4, 0.6), "loudness": (0.4, 0.6),
        "speechiness": (0.03, 0.2), "acousticness": (0.3, 0.6), "instrumentalness": (0.2, 0.6),
        "liveness": (0.1, 0.4), "valence": (0.4, 0.7), "tempo": (0.4, 0.7), "spec_rate": (0.5, 0.7)
    },
    "Romantic": {
        "danceability": (0.4, 0.7), "energy": (0.4, 0.65), "loudness": (0.5, 0.75),
        "speechiness": (0.03, 0.25), "acousticness": (0.5, 0.8), "instrumentalness": (0.0, 0.3),
        "liveness": (0.1, 0.4), "valence": (0.6, 0.9), "tempo": (0.4, 0.6), "spec_rate": (0.4, 0.6)
    },
    "Chill": {
        "danceability": (0.5, 0.8), "energy": (0.3, 0.55), "loudness": (0.3, 0.6),
        "speechiness": (0.04, 0.2), "acousticness": (0.4, 0.7), "instrumentalness": (0.2, 0.6),
        "liveness": (0.2, 0.5), "valence": (0.5, 0.8), "tempo": (0.3, 0.6), "spec_rate": (0.4, 0.7)
    },
    "Determined": {
        "danceability": (0.5, 0.7), "energy": (0.7, 0.9), "loudness": (0.7, 0.9),
        "speechiness": (0.05, 0.25), "acousticness": (0.1, 0.4), "instrumentalness": (0.0, 0.2),
        "liveness": (0.2, 0.5), "valence": (0.4, 0.6), "tempo": (0.7, 0.9), "spec_rate": (0.6, 0.8)
    },
    "Reflective": {
        "danceability": (0.3, 0.6), "energy": (0.3, 0.5), "loudness": (0.3, 0.6),
        "speechiness": (0.03, 0.15), "acousticness": (0.6, 0.9), "instrumentalness": (0.1, 0.5),
        "liveness": (0.2, 0.5), "valence": (0.3, 0.6), "tempo": (0.4, 0.6), "spec_rate": (0.4, 0.6)
    },
    "Confident": {
        "danceability": (0.6, 0.85), "energy": (0.7, 0.9), "loudness": (0.8, 1.0),
        "speechiness": (0.05, 0.25), "acousticness": (0.1, 0.4), "instrumentalness": (0.0, 0.2),
        "liveness": (0.2, 0.5), "valence": (0.6, 0.9), "tempo": (0.6, 0.9), "spec_rate": (0.6, 0.9)
    },
    "Anxious": {
        "danceability": (0.3, 0.5), "energy": (0.5, 0.7), "loudness": (0.5, 0.7),
        "speechiness": (0.05, 0.3), "acousticness": (0.2, 0.5), "instrumentalness": (0.1, 0.4),
        "liveness": (0.4, 0.7), "valence": (0.2, 0.5), "tempo": (0.5, 0.7), "spec_rate": (0.5, 0.7)
    },
    "Excited": {
        "danceability": (0.7, 0.9), "energy": (0.8, 1.0), "loudness": (0.8, 1.0),
        "speechiness": (0.05, 0.2), "acousticness": (0.1, 0.3), "instrumentalness": (0.0, 0.1),
        "liveness": (0.3, 0.6), "valence": (0.7, 1.0), "tempo": (0.8, 1.0), "spec_rate": (0.7, 1.0)
    }
}

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "spec_rate"
]

CSV_PATH = "/Users/aryansaxena/Desktop/Mood_model_Project/data/SpotifyAudioFeaturesNov2018.csv"
OUTPUT_PATH = "mood_dataset_12mood.csv"
SIMILARITY_THRESHOLD = 0.70

def calculate_similarity(row, profile):
    score = 0.0
    count = 0
    for feat, (min_val, max_val) in profile.items():
        if feat == 'spec_rate':
            val = min(max(float(row.get("tempo", 0)) / 200.0, 0.0), 1.0)
        elif feat == 'tempo':
            # normalize tempo (40-200) to 0-1
            val = max(0.0, min(1.0, (float(row.get('tempo', 120)) - 40.0) / 160.0))
        elif feat == 'loudness':
            # normalize loudness (-60 to 0) to 0-1
            val = max(0.0, min(1.0, (float(row.get('loudness', -8)) + 60.0) / 60.0))
        else:
            val = float(row.get(feat, 0.5))
            
        if min_val <= val <= max_val:
            sim = 1.0
        else:
            dist = min(abs(val - min_val), abs(val - max_val))
            sim = max(0.0, 1.0 - dist)
        score += sim
        count += 1
    return score / count if count > 0 else 0.0

def main():
    print(f"Loading raw Spotify dataset: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} tracks. Ranking similarities across all 12 moods...")
    
    # Calculate similarities for all songs and all moods
    all_pairs = []
    # Using itertuples is much faster than iterrows
    for row in tqdm(df.itertuples(), total=len(df)):
        row_dict = row._asdict()
        for mood, profile in MOOD_PROFILES.items():
            score = calculate_similarity(row_dict, profile)
            all_pairs.append((row.Index, mood, score))
            
    # Sort pairs by score descending
    print("Sorting similarity pairs...")
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Greedy balanced assignment
    target_size = 2500  # Expand to 2500 for a larger, richer dataset
    assigned_songs = set()
    class_counts = {mood: 0 for mood in MOOD_PROFILES.keys()}
    selected_indices = []
    selected_labels = []
    min_scores = {mood: 1.0 for mood in MOOD_PROFILES.keys()}
    
    print(f"Selecting exactly {target_size} balanced tracks per mood...")
    for song_idx, mood, score in all_pairs:
        if song_idx not in assigned_songs and class_counts[mood] < target_size:
            assigned_songs.add(song_idx)
            class_counts[mood] += 1
            selected_indices.append(song_idx)
            selected_labels.append(mood)
            min_scores[mood] = min(min_scores[mood], score)
            
    # Build final DataFrame
    selected_df = df.loc[selected_indices].copy()
    selected_df["labels"] = selected_labels
    selected_df["spec_rate"] = selected_df["tempo"].apply(lambda t: min(max(float(t) / 200.0, 0.0), 1.0))
    
    # Shuffle
    final_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nBalanced Mood Distribution:")
    print(final_df["labels"].value_counts())
    print("\nMinimum similarity score per class:")
    for mood, min_score in min_scores.items():
        print(f"  {mood}: {min_score:.4f}")
        
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved balanced dataset to: {OUTPUT_PATH} (Total tracks: {len(final_df):,})")

if __name__ == "__main__":
    main()
