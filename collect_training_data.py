"""
Collect labeled training data from Spotify for mood classification.

This script helps you create a real dataset by:
1. Fetching tracks from mood-labeled playlists
2. Getting audio features for those tracks
3. Creating a labeled dataset for training

Usage:
    python collect_training_data.py --output mood_dataset.csv
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import time


# Load environment variables
load_dotenv()

# Mood-labeled playlist seeds
# You can find these by searching "happy playlist", "sad songs", etc. on Spotify
MOOD_PLAYLISTS = {
    "Happy": [
        "37i9dQZF1DXdPec7aLTmlC",  # Happy Hits
        "37i9dQZF1DX3rxVfibe1L0",  # Mood Booster
        "37i9dQZF1DX7KNKjOK0o75",  # Happy Pop
    ],
    "Sad": [
        "37i9dQZF1DX3YSRoSdA634",  # Life Sucks
        "37i9dQZF1DX7qK8ma5wgG1",  # Sad Songs
        "37i9dQZF1DWSqBruwoIXkA",  # Sad Indie
    ],
    "Calm": [
        "37i9dQZF1DWZd79rJ6a7lp",  # Peaceful Piano
        "37i9dQZF1DX4sWSpwq3LiO",  # Peaceful Guitar
        "37i9dQZF1DX3Ogo9pFvBkY",  # Calm Vibes
    ],
    "Energetic": [
        "37i9dQZF1DX76Wlfdnj7AP",  # Beast Mode
        "37i9dQZF1DX0HRj9P7NxeE",  # Power Workout
        "37i9dQZF1DXdxcBWuJkbcy",  # Power Hour
    ],
    "Angry": [
        "37i9dQZF1DX1tyCD9QhIWF",  # Rage Beats
        "37i9dQZF1DWWv1zzAOcVaX",  # Angry Metal
        "37i9dQZF1DXa3c5GDfLqvW",  # Aggressive Rap
    ],
    "Focus": [
        "37i9dQZF1DX8NTLI2TtZa6",  # Deep Focus
        "37i9dQZF1DWZeKCadgRdKQ",  # Focus Flow
        "37i9dQZF1DX3PFzdbtx1Us",  # Instrumental Study
    ]
}


def get_spotify_client():
    """
    Initialize Spotify client with credentials from environment.
    """
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env file"
        )
    
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    
    return spotipy.Spotify(auth_manager=auth_manager)


def get_playlist_tracks(sp, playlist_id, mood_label):
    """
    Get all tracks from a playlist with their mood label.
    
    Args:
        sp: Spotify client
        playlist_id: Spotify playlist ID
        mood_label: Mood label for this playlist
    
    Returns:
        List of track IDs with mood labels
    """
    tracks = []
    offset = 0
    limit = 100
    
    while True:
        try:
            results = sp.playlist_tracks(
                playlist_id,
                offset=offset,
                limit=limit,
                fields='items(track(id,name,artists)),next'
            )
            
            for item in results['items']:
                if item['track'] and item['track']['id']:
                    tracks.append({
                        'track_id': item['track']['id'],
                        'mood': mood_label
                    })
            
            if not results['next']:
                break
            
            offset += limit
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching playlist {playlist_id}: {e}")
            break
    
    return tracks


def get_audio_features_batch(sp, track_ids):
    """
    Get audio features for a batch of tracks.
    
    Args:
        sp: Spotify client
        track_ids: List of track IDs
    
    Returns:
        List of audio feature dictionaries
    """
    features = []
    
    # Spotify allows max 100 tracks per request
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        
        try:
            batch_features = sp.audio_features(batch)
            features.extend([f for f in batch_features if f is not None])
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error fetching audio features: {e}")
            continue
    
    return features


def collect_dataset(sp, max_tracks_per_mood=1000):
    """
    Collect complete dataset from mood playlists.
    
    Args:
        sp: Spotify client
        max_tracks_per_mood: Maximum tracks to collect per mood
    
    Returns:
        pandas DataFrame with audio features and mood labels
    """
    all_tracks = []
    
    print("\n🎵 Collecting tracks from mood playlists...")
    
    for mood, playlist_ids in MOOD_PLAYLISTS.items():
        print(f"\n📂 Collecting {mood} tracks...")
        mood_tracks = []
        
        for playlist_id in tqdm(playlist_ids, desc=f"{mood} playlists"):
            tracks = get_playlist_tracks(sp, playlist_id, mood)
            mood_tracks.extend(tracks)
            
            if len(mood_tracks) >= max_tracks_per_mood:
                break
        
        # Remove duplicates
        unique_tracks = {t['track_id']: t for t in mood_tracks}.values()
        mood_tracks = list(unique_tracks)[:max_tracks_per_mood]
        
        print(f"✅ Collected {len(mood_tracks)} unique {mood} tracks")
        all_tracks.extend(mood_tracks)
    
    print(f"\n📊 Total unique tracks: {len(all_tracks)}")
    
    # Get audio features
    print("\n🎹 Fetching audio features...")
    track_ids = [t['track_id'] for t in all_tracks]
    
    audio_features = []
    for i in tqdm(range(0, len(track_ids), 100), desc="Fetching features"):
        batch = track_ids[i:i+100]
        batch_features = get_audio_features_batch(sp, batch)
        audio_features.extend(batch_features)
    
    print(f"✅ Retrieved features for {len(audio_features)} tracks")
    
    # Create mood mapping
    mood_map = {t['track_id']: t['mood'] for t in all_tracks}
    
    # Build dataset
    dataset = []
    for features in audio_features:
        if features and features['id'] in mood_map:
            dataset.append({
                'track_id': features['id'],
                'valence': features['valence'],
                'energy': features['energy'],
                'danceability': features['danceability'],
                'acousticness': features['acousticness'],
                'instrumentalness': features['instrumentalness'],
                'speechiness': features['speechiness'],
                'tempo': features['tempo'],
                'loudness': features['loudness'],
                'liveness': features['liveness'],
                'key': features['key'],
                'mode': features['mode'],
                'time_signature': features['time_signature'],
                'mood': mood_map[features['id']]
            })
    
    df = pd.DataFrame(dataset)
    
    return df


def analyze_dataset(df):
    """
    Analyze and display dataset statistics.
    """
    print("\n" + "=" * 60)
    print("📊 Dataset Statistics")
    print("=" * 60)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"\nMood distribution:")
    print(df['mood'].value_counts())
    
    print(f"\nFeature statistics:")
    print(df.describe())
    
    # Check for class imbalance
    mood_counts = df['mood'].value_counts()
    min_count = mood_counts.min()
    max_count = mood_counts.max()
    imbalance_ratio = max_count / min_count
    
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2:
        print("⚠️  WARNING: Significant class imbalance detected!")
        print("   Consider collecting more data for underrepresented moods.")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values detected:")
        print(missing[missing > 0])


def balance_dataset(df, method='undersample'):
    """
    Balance the dataset to have equal samples per class.
    
    Args:
        df: Input DataFrame
        method: 'undersample' or 'oversample'
    
    Returns:
        Balanced DataFrame
    """
    mood_counts = df['mood'].value_counts()
    
    if method == 'undersample':
        # Use the minimum count
        target_count = mood_counts.min()
        balanced_dfs = []
        
        for mood in df['mood'].unique():
            mood_df = df[df['mood'] == mood].sample(n=target_count, random_state=42)
            balanced_dfs.append(mood_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif method == 'oversample':
        # Use the maximum count
        target_count = mood_counts.max()
        balanced_dfs = []
        
        for mood in df['mood'].unique():
            mood_df = df[df['mood'] == mood]
            if len(mood_df) < target_count:
                # Oversample with replacement
                mood_df = mood_df.sample(n=target_count, replace=True, random_state=42)
            balanced_dfs.append(mood_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    else:
        raise ValueError("Method must be 'undersample' or 'oversample'")
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Dataset balanced using {method}")
    print(f"   Original size: {len(df)}")
    print(f"   Balanced size: {len(balanced_df)}")
    print(f"\nNew mood distribution:")
    print(balanced_df['mood'].value_counts())
    
    return balanced_df


def main():
    """
    Main data collection pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Collect training data from Spotify for mood classification'
    )
    parser.add_argument('--output', type=str, default='mood_dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--max-per-mood', type=int, default=1000,
                        help='Maximum tracks to collect per mood')
    parser.add_argument('--balance', type=str, choices=['undersample', 'oversample', 'none'],
                        default='undersample',
                        help='Balancing method for class distribution')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎵 Moodify-AI Training Data Collection")
    print("=" * 60)
    
    # Initialize Spotify client
    print("\n🔑 Authenticating with Spotify...")
    sp = get_spotify_client()
    print("✅ Authentication successful")
    
    # Collect dataset
    df = collect_dataset(sp, max_tracks_per_mood=args.max_per_mood)
    
    # Analyze dataset
    analyze_dataset(df)
    
    # Balance dataset if requested
    if args.balance != 'none':
        df = balance_dataset(df, method=args.balance)
    
    # Save dataset
    df.to_csv(args.output, index=False)
    print(f"\n✅ Dataset saved to {args.output}")
    
    print("\n" + "=" * 60)
    print("✅ Data collection completed successfully!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review the dataset: {args.output}")
    print(f"  2. Train the model: python train_mood_model.py --dataset {args.output}")
    print(f"  3. Deploy the model to your ML service")


if __name__ == '__main__':
    main()