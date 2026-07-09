"""
Prepare MoodiQ Dataset for Model Training
------------------------------------------
Converts the downloaded MoodiQ dataset to match the format expected by train_mood_model.py

Required columns in output:
- valence, energy, danceability, acousticness, instrumentalness, speechiness
- tempo, loudness, liveness, key, mode, time_signature
- mood (one of: Happy, Sad, Calm, Energetic, Angry, Focus)

Usage:
    python prepare_MoodiQ_dataset.py --input MoodiQ_dataset.csv --output mood_dataset.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Expected audio features for model training
REQUIRED_FEATURES = [
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

# Target mood classes
TARGET_MOODS = ["Happy", "Sad", "Calm", "Energetic", "Angry", "Focus"]

# Numeric label mapping for MoodiQ dataset
# Based on the dataset: 0=Sad, 1=Happy, 2=Energetic, 3=Calm
NUMERIC_MOOD_MAPPINGS = {
    0: 'Sad',
    1: 'Happy', 
    2: 'Energetic',
    3: 'Calm'
}

# Text mood mapping dictionary (for text-based labels)
MOOD_MAPPINGS = {
    # Common variations to our target moods
    'happy': 'Happy',
    'joy': 'Happy',
    'joyful': 'Happy',
    'cheerful': 'Happy',
    'upbeat': 'Happy',
    
    'sad': 'Sad',
    'sadness': 'Sad',
    'melancholy': 'Sad',
    'depressed': 'Sad',
    'blue': 'Sad',
    'sorrowful': 'Sad',
    
    'calm': 'Calm',
    'peaceful': 'Calm',
    'relaxed': 'Calm',
    'chill': 'Calm',
    'serene': 'Calm',
    'tranquil': 'Calm',
    
    'energetic': 'Energetic',
    'energy': 'Energetic',
    'excited': 'Energetic',
    'uplifting': 'Energetic',
    'pump up': 'Energetic',
    
    'angry': 'Angry',
    'anger': 'Angry',
    'rage': 'Angry',
    'aggressive': 'Angry',
    'mad': 'Angry',
    
    'focus': 'Focus',
    'focused': 'Focus',
    'concentration': 'Focus',
    'study': 'Focus',
    'work': 'Focus'
}

def find_mood_column(df):
    
    possible_names = ['mood', 'emotion', 'label', 'labels', 'class', 'sentiment', 'feeling']
    
    for col in df.columns:
        if col.lower() in possible_names:
            return col
    
    # Check for columns with these words
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in ['mood', 'emotion', 'label']):
            return col
    
    return None

def map_mood(original_mood):
    
    if pd.isna(original_mood):
        return None
    
    # Handle numeric labels
    if isinstance(original_mood, (int, float, np.integer, np.floating)):
        return NUMERIC_MOOD_MAPPINGS.get(int(original_mood))
    
    mood_lower = str(original_mood).lower().strip()
    
    # Direct match
    if mood_lower in [m.lower() for m in TARGET_MOODS]:
        return mood_lower.capitalize()
    
    # Try mapping
    if mood_lower in MOOD_MAPPINGS:
        return MOOD_MAPPINGS[mood_lower]
    
    # Try partial match
    for key, value in MOOD_MAPPINGS.items():
        if key in mood_lower:
            return value
    
    return None

def prepare_dataset(input_path, output_path, max_per_mood=None, balance=True):
    
    print("=" * 60)
    print("🎵 MoodiQ Dataset Preparation")
    print("=" * 60)
    
    # Check if file exists
    if not Path(input_path).exists():
        print(f"❌ File not found: {input_path}")
        print("\n💡 Please update the path to your downloaded MoodiQ dataset")
        return
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df):,} rows")
    
    # Show columns
    print(f"\n📋 Available columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Find mood column
    mood_col = find_mood_column(df)
    if not mood_col:
        print(f"\n❌ Could not find mood/emotion column!")
        print("Available columns:", df.columns.tolist())
        print("\n💡 Please manually specify the mood column name")
        return
    
    print(f"\n🎭 Found mood column: '{mood_col}'")
    print(f"\nOriginal mood distribution:")
    print(df[mood_col].value_counts())
    
    # Check if labels are numeric
    unique_labels = df[mood_col].unique()
    is_numeric = all(isinstance(x, (int, float, np.integer, np.floating)) for x in unique_labels if pd.notna(x))
    
    if is_numeric:
        print(f"\n🔢 Detected numeric labels: {sorted([int(x) for x in unique_labels if pd.notna(x)])}")
        print(f"💡 Using mapping: {NUMERIC_MOOD_MAPPINGS}")
        print(f"⚠️  If this mapping is incorrect, update NUMERIC_MOOD_MAPPINGS in the script")
    
    # Check for required audio features
    print(f"\n🎼 Checking for required audio features...")
    missing_features = []
    available_features = []
    
    for feature in REQUIRED_FEATURES:
        if feature in df.columns:
            available_features.append(feature)
            print(f"  ✅ {feature}")
        else:
            missing_features.append(feature)
    
    # Handle missing features by generating defaults
    if missing_features:
        print(f"\n⚠️ Missing features will be filled with defaults:")
        for feature in missing_features:
            if feature == 'key':
                df['key'] = np.random.randint(0, 12, size=len(df))
                print(f"  📝 {feature} - generated (random 0-11)")
            elif feature == 'mode':
                df['mode'] = np.random.randint(0, 2, size=len(df))
                print(f"  📝 {feature} - generated (random 0-1)")
            elif feature == 'time_signature':
                df['time_signature'] = 4  # Most common
                print(f"  📝 {feature} - set to 4 (most common)")
            available_features.append(feature)
    
    if len(available_features) < 8:
        print(f"\n❌ Only {len(available_features)}/{len(REQUIRED_FEATURES)} features available")
        print("❌ Dataset may not have enough features for training")
        return
    
    # Map moods to target classes
    print(f"\n🔄 Mapping moods to target classes...")
    df['mapped_mood'] = df[mood_col].apply(map_mood)
    
    # Remove unmapped moods
    before_count = len(df)
    df = df[df['mapped_mood'].notna()]
    after_count = len(df)
    
    if after_count == 0:
        print(f"\n❌ No moods could be mapped to target classes!")
        print("💡 You may need to update MOOD_MAPPINGS or NUMERIC_MOOD_MAPPINGS dictionary")
        print(f"   Original moods: {df[mood_col].unique().tolist()}")
        return
    
    print(f"✅ Mapped {after_count:,} songs (removed {before_count - after_count:,} unmapped)")
    
    print(f"\n📊 Mapped mood distribution:")
    print(df['mapped_mood'].value_counts())
    
    # Select only required columns
    output_columns = available_features + ['mapped_mood']
    
    # Add optional metadata columns if available
    for col in ['track_name', 'track_id', 'artist', 'artist_name']:
        if col in df.columns:
            output_columns.insert(0, col)
    
    df_clean = df[output_columns].copy()
    df_clean.rename(columns={'mapped_mood': 'mood'}, inplace=True)
    
    # Remove any rows with missing audio features
    before_count = len(df_clean)
    df_clean = df_clean.dropna(subset=available_features)
    after_count = len(df_clean)
    
    if before_count != after_count:
        print(f"\n🧹 Removed {before_count - after_count:,} rows with missing features")
    
    # Balance dataset if requested
    if balance:
        print(f"\n⚖️ Balancing dataset...")
        mood_counts = df_clean['mood'].value_counts()
        available_moods = mood_counts.index.tolist()
        min_count = mood_counts.min()
        
        if max_per_mood:
            min_count = min(min_count, max_per_mood)
        
        balanced_dfs = []
        for mood in available_moods:
            mood_df = df_clean[df_clean['mood'] == mood]
            if len(mood_df) > 0:
                if len(mood_df) > min_count:
                    mood_df = mood_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(mood_df)
        
        df_clean = pd.concat(balanced_dfs, ignore_index=True)
        df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Balanced to {min_count:,} samples per mood")
    
    # Final mood distribution
    print(f"\n📊 Final mood distribution:")
    final_counts = df_clean['mood'].value_counts()
    for mood, count in final_counts.items():
        print(f"  {mood}: {count:,}")
    
    # Save dataset
    df_clean.to_csv(output_path, index=False)
    print(f"\n💾 Saved processed dataset → {output_path}")
    print(f"   Total samples: {len(df_clean):,}")
    print(f"   Features: {len(available_features)}")
    
    # Verify format compatibility
    print(f"\n✅ Dataset is ready for train_mood_model.py!")
    print(f"\n🚀 Next step:")
    print(f"   python train_mood_model.py --dataset {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation completed!")
    print("=" * 60)
    
    return df_clean

def main():
    parser = argparse.ArgumentParser(
        description="Prepare MoodiQ dataset for mood classification training"
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='MoodiQ_dataset.csv',
        help='Path to input MoodiQ dataset CSV'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='mood_dataset.csv',
        help='Path to save processed dataset'
    )
    parser.add_argument(
        '--max-per-mood',
        type=int,
        default=None,
        help='Maximum samples per mood (default: use all available)'
    )
    parser.add_argument(
        '--no-balance',
        action='store_true',
        help='Skip dataset balancing'
    )
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_path=args.input,
        output_path=args.output,
        max_per_mood=args.max_per_mood,
        balance=not args.no_balance
    )

if __name__ == '__main__':
    main()