#!/usr/bin/env python3
"""
label_12mood_v2.py - Proper mood labeling using Russell's Circumplex Model
============================================================================
Uses the same priority-rule engine as production model_service.py so the
model learns what the rules encode instead of circular label noise.

Usage:  python label_12mood_v2.py
Output: mood_dataset_v2.csv
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT_CSV  = "/Users/aryansaxena/Desktop/Mood_model_Project/data/SpotifyAudioFeaturesNov2018.csv"
FALLBACK_CSV = "mood_dataset_12mood.csv"
OUTPUT_CSV = "mood_dataset_v2.csv"
TARGET_PER_CLASS = 2500  # 2500 x 12 = 30,000

def label_row(row) -> str:
    v   = float(row.get('valence',          0.5))
    e   = float(row.get('energy',           0.5))
    d   = float(row.get('danceability',     0.5))
    a   = float(row.get('acousticness',     0.5))
    ins = float(row.get('instrumentalness', 0.0))
    raw_t = float(row.get('tempo', 120))
    t   = max(0.0, min(1.0, (raw_t - 40) / 160))
    raw_l = float(row.get('loudness', -8))
    loud  = max(0.0, min(1.0, (raw_l + 60) / 60))

    if v > 0.7  and e > 0.78 and d > 0.65:                   return 'Excited'
    if e > 0.80 and v < 0.45:                                  return 'Determined'
    if e > 0.72 and v > 0.30:                                  return 'Energetic'
    if e > 0.50 and v < 0.35:                                  return 'Anxious'
    if e > 0.65 and v > 0.6 and loud > 0.75 and d < 0.75:     return 'Confident'
    if v < 0.30 and e < 0.55 and ins < 0.60:                  return 'Sad'
    if v > 0.72 and e > 0.50 and d > 0.65:                    return 'Happy'
    if v > 0.55 and e < 0.55 and a > 0.45 and ins < 0.4:      return 'Romantic'
    if a > 0.65 and ins > 0.50 and e < 0.45:                  return 'Reflective'
    if ins > 0.60 and e < 0.55 and a < 0.70:                  return 'Focused'
    if e < 0.42 and v > 0.30 and (a > 0.35 or ins > 0.40):   return 'Calm'
    if d > 0.60 and e < 0.58 and v > 0.45:                    return 'Chill'
    if a > 0.60 and t < 0.50 and 0.25 < v < 0.70:            return 'Reflective'
    if v >= 0.5 and e >= 0.5:  return 'Happy'
    if v >= 0.5 and e <  0.5:  return 'Chill'
    if v <  0.5 and e >= 0.5:  return 'Anxious'
    return 'Sad'

def main():
    if os.path.exists(INPUT_CSV):
        print(f"Loading: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
    elif os.path.exists(FALLBACK_CSV):
        print(f"Using fallback: {FALLBACK_CSV}")
        df = pd.read_csv(FALLBACK_CSV)
        if 'labels' in df.columns:
            df = df.drop(columns=['labels'])
    else:
        print("ERROR: No input CSV found."); return

    print(f"Loaded {len(df):,} rows. Labeling...")
    FEATS = ['valence','energy','danceability','acousticness','instrumentalness','speechiness','tempo','loudness']
    for f in FEATS:
        if f not in df.columns: df[f] = 0.5
    df = df.dropna(subset=FEATS).reset_index(drop=True)

    labels = [label_row(row) for _, row in tqdm(df.iterrows(), total=len(df))]
    df['labels'] = labels
    df['spec_rate'] = df['tempo'].apply(lambda t: min(max(float(t) / 200.0, 0.0), 1.0))

    print("\nRaw distribution:")
    counts = df['labels'].value_counts()
    for mood, cnt in counts.items():
        print(f"  {mood:<15}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    print(f"\nBalancing to {TARGET_PER_CLASS}/class...")
    parts = []
    for mood in counts.index:
        sub = df[df['labels'] == mood]
        n = len(sub)
        if n >= TARGET_PER_CLASS:
            parts.append(sub.sample(n=TARGET_PER_CLASS, random_state=42))
        else:
            print(f"  ⚠️  {mood}: {n} samples, over-sampling...")
            parts.append(sub.sample(n=TARGET_PER_CLASS, replace=True, random_state=42))

    out = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(out):,} rows → {OUTPUT_CSV}")
    print("Next: run train_mood_model_12mood.py with csv_path = 'mood_dataset_v2.csv'")

if __name__ == "__main__":
    main()
