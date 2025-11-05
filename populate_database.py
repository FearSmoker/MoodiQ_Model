"""
Spotify Dataset to MongoDB Populator with MoodiQ ML Model Integration
=====================================================================

This script:
1. Loads Spotify dataset CSVs (2018 and 2019)
2. Extracts the 12 required audio features
3. Uses your ONNX ML model to predict moods (1-3 tags per song)
4. Populates MongoDB 'databasesongs' collection

Requirements:
    pip install pandas pymongo motor asyncio python-dotenv onnxruntime numpy tqdm

Usage:
    python populate_database.py --csv2018 spotify_2018.csv --csv2019 spotify_2019.csv --batch-size 500
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import signal

import pandas as pd
import numpy as np
import onnxruntime as ort
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import json

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "moodiq"
COLLECTION_NAME = "databasesongs"

# Model Configuration
MODEL_PATH = "models/mood_model.onnx"
METADATA_PATH = "models/model_metadata.json"

# Required 12 audio features (matching your model)
REQUIRED_FEATURES = [
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

# Base mood classes from your trained model
BASE_MOOD_CLASSES = ["Calm", "Energetic", "Happy", "Sad"]

# Extended 12 moods with profiles (from your model_service.py)
EXTENDED_MOODS = {
    "Relaxed": {
        "base_moods": ["Calm"],
        "profile": {"valence": 0.5, "energy": 0.25, "danceability": 0.3},
        "weights": {"valence": 0.8, "energy": 1.0, "acousticness": 0.9}
    },
    "Focused": {
        "base_moods": ["Calm"],
        "profile": {"valence": 0.45, "energy": 0.4, "danceability": 0.35},
        "weights": {"energy": 1.0, "speechiness": 0.9, "instrumentalness": 0.8}
    },
    "Romantic": {
        "base_moods": ["Calm", "Happy"],
        "profile": {"valence": 0.6, "energy": 0.35, "danceability": 0.4},
        "weights": {"valence": 0.9, "energy": 0.8, "acousticness": 0.7}
    },
    "Excited": {
        "base_moods": ["Happy", "Energetic"],
        "profile": {"valence": 0.8, "energy": 0.85, "danceability": 0.8},
        "weights": {"valence": 0.9, "energy": 1.0, "danceability": 0.9}
    },
    "Angry": {
        "base_moods": ["Energetic", "Sad"],
        "profile": {"valence": 0.25, "energy": 0.85, "danceability": 0.5},
        "weights": {"valence": 1.0, "energy": 1.0, "loudness": 0.9}
    },
    "Chill": {
        "base_moods": ["Calm", "Happy"],
        "profile": {"valence": 0.6, "energy": 0.3, "danceability": 0.45},
        "weights": {"energy": 1.0, "valence": 0.8, "danceability": 0.7}
    },
    "Melancholic": {
        "base_moods": ["Sad"],
        "profile": {"valence": 0.2, "energy": 0.25, "danceability": 0.3},
        "weights": {"valence": 1.0, "energy": 0.9, "acousticness": 0.8}
    },
    "Dreamy": {
        "base_moods": ["Calm", "Sad"],
        "profile": {"valence": 0.4, "energy": 0.3, "danceability": 0.35},
        "weights": {"energy": 0.9, "acousticness": 1.0, "instrumentalness": 0.8}
    },
    "Motivated": {
        "base_moods": ["Energetic"],
        "profile": {"valence": 0.65, "energy": 0.75, "danceability": 0.65},
        "weights": {"energy": 1.0, "valence": 0.8, "tempo": 0.7}
    },
    "Joyful": {
        "base_moods": ["Happy"],
        "profile": {"valence": 0.85, "energy": 0.7, "danceability": 0.75},
        "weights": {"valence": 1.0, "energy": 0.8, "danceability": 0.9}
    },
    "Ambient": {
        "base_moods": ["Calm"],
        "profile": {"valence": 0.5, "energy": 0.2, "danceability": 0.25},
        "weights": {"instrumentalness": 1.0, "energy": 0.9, "speechiness": 0.8}
    },
    "Party": {
        "base_moods": ["Energetic", "Happy"],
        "profile": {"valence": 0.8, "energy": 0.9, "danceability": 0.9},
        "weights": {"danceability": 1.0, "energy": 1.0, "valence": 0.9}
    }
}

ALL_MOOD_LABELS = list(EXTENDED_MOODS.keys())

# ============================================
# MODEL LOADER
# ============================================

class MoodPredictor:
    """ONNX Model wrapper for mood prediction"""
    
    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.session = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.mood_classes = BASE_MOOD_CLASSES
        
    def load(self):
        """Load ONNX model and metadata"""
        print(f"📦 Loading ONNX model from {self.model_path}...")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load ONNX model
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            self.model_path,
            session_options,
            providers=['CPUExecutionProvider']
        )
        
        print("✅ ONNX model loaded")
        
        # Load metadata
        if Path(self.metadata_path).exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.scaler_mean = np.array(metadata['scaler_mean'], dtype=np.float32)
                self.scaler_scale = np.array(metadata['scaler_scale'], dtype=np.float32)
                self.mood_classes = metadata.get('mood_classes', BASE_MOOD_CLASSES)
                print("✅ Metadata loaded")
        else:
            print("⚠️  Metadata not found, using defaults")
            self.scaler_mean = None
            self.scaler_scale = None
    
    def normalize_features(self, features: Dict) -> np.ndarray:
        """Normalize audio features using saved scaler"""
        feature_values = []
        for feature_name in REQUIRED_FEATURES:
            value = features.get(feature_name, 0.5)
            feature_values.append(float(value))
        
        feature_array = np.array(feature_values, dtype=np.float32)
        
        if self.scaler_mean is not None and self.scaler_scale is not None:
            feature_array = (feature_array - self.scaler_mean) / self.scaler_scale
            feature_array = feature_array.astype(np.float32)
        
        return feature_array
    
    def calculate_mood_similarity(self, features: Dict, mood_name: str) -> float:
        """Calculate similarity between track features and mood profile"""
        mood_profile = EXTENDED_MOODS[mood_name]["profile"]
        weights = EXTENDED_MOODS[mood_name]["weights"]
        
        total_similarity = 0.0
        total_weight = 0.0
        
        for feature, target_value in mood_profile.items():
            if feature not in features:
                continue
            
            actual_value = features[feature]
            weight = weights.get(feature, 0.5)
            
            # Normalize feature values
            if feature == 'tempo':
                actual_normalized = (actual_value - 40) / 160
                target_normalized = (target_value - 40) / 160
            elif feature == 'loudness':
                actual_normalized = (actual_value + 60) / 60
                target_normalized = (target_value + 60) / 60
            else:
                actual_normalized = actual_value
                target_normalized = target_value
            
            distance = abs(actual_normalized - target_normalized)
            similarity = 1.0 - distance
            
            total_similarity += similarity * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_similarity / total_weight
        
        return 0.0
    
    def get_multi_mood_tags(
        self, 
        features: Dict, 
        min_similarity: float = 0.70,
        max_tags: int = 3
    ) -> List[tuple]:
        """Get 1-3 mood tags based on feature similarity"""
        mood_scores = []
        
        for mood_name in ALL_MOOD_LABELS:
            similarity = self.calculate_mood_similarity(features, mood_name)
            
            if similarity >= min_similarity:
                mood_scores.append((mood_name, similarity))
        
        mood_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure at least 1 tag, max 3 tags
        if not mood_scores:
            # If no moods meet threshold, use base model prediction
            return []
        
        return mood_scores[:max_tags]
    
    def predict_base_mood(self, features: Dict) -> tuple:
        """Predict base mood using ONNX model"""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Normalize features
        input_data = self.normalize_features(features)
        input_data = input_data.reshape(1, -1).astype(np.float32)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        
        probabilities = outputs[0][0]
        predicted_index = np.argmax(probabilities)
        confidence = float(probabilities[predicted_index])
        base_mood = self.mood_classes[predicted_index]
        
        return base_mood, confidence
    
    def predict(self, features: Dict) -> Dict:
        """Full prediction with multi-mood tags"""
        # Get base mood from model
        base_mood, base_confidence = self.predict_base_mood(features)
        
        # Get multi-mood tags
        mood_tags = self.get_multi_mood_tags(features, min_similarity=0.70, max_tags=3)
        
        # If no tags meet threshold, use base mood
        if not mood_tags:
            mood_tags = [(base_mood, base_confidence)]
        
        # Extract mood names and scores
        primary_mood = mood_tags[0][0]
        all_moods = [mood for mood, _ in mood_tags]
        mood_scores = {mood: float(score) for mood, score in mood_tags}
        
        return {
            'base_mood': base_mood,
            'base_confidence': base_confidence,
            'primary_mood': primary_mood,
            'all_moods': all_moods,
            'mood_scores': mood_scores,
            'num_tags': len(all_moods)
        }


# ============================================
# DATA LOADER
# ============================================

def load_spotify_dataset(csv_files: List[str]) -> pd.DataFrame:
    """Load and combine Spotify CSV files"""
    print(f"\n📂 Loading Spotify dataset from {len(csv_files)} file(s)...")
    
    dataframes = []
    
    for csv_file in csv_files:
        if not Path(csv_file).exists():
            print(f"⚠️  File not found: {csv_file}")
            continue
        
        print(f"   Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"   ✅ Loaded {len(df):,} rows")
        dataframes.append(df)
    
    if not dataframes:
        raise FileNotFoundError("No valid CSV files found")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"\n✅ Total rows loaded: {len(combined_df):,}")
    
    return combined_df


def extract_required_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the 12 required audio features"""
    print("\n🔍 Extracting required audio features...")
    
    # Check which features are available
    available_features = []
    missing_features = []
    
    for feature in REQUIRED_FEATURES:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"✅ Available features: {len(available_features)}/{len(REQUIRED_FEATURES)}")
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        raise ValueError(f"Dataset missing required features: {missing_features}")
    
    # Extract features + metadata columns
    metadata_columns = []
    for col in ['id', 'name', 'artists', 'artist', 'track_name', 'artist_name', 'year', 'release_date', 'popularity', 'duration_ms']:
        if col in df.columns:
            metadata_columns.append(col)
    
    selected_columns = metadata_columns + available_features
    df_filtered = df[selected_columns].copy()
    
    # Remove rows with missing audio features
    before_count = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=available_features)
    after_count = len(df_filtered)
    
    if before_count != after_count:
        print(f"🧹 Removed {before_count - after_count:,} rows with missing features")
    
    # Remove duplicates
    if 'id' in df_filtered.columns:
        before_count = len(df_filtered)
        df_filtered = df_filtered.drop_duplicates(subset=['id'])
        after_count = len(df_filtered)
        if before_count != after_count:
            print(f"🧹 Removed {before_count - after_count:,} duplicate tracks")
    
    print(f"✅ Final dataset: {len(df_filtered):,} tracks")
    
    return df_filtered


# ============================================
# DATABASE OPERATIONS
# ============================================

class DatabasePopulator:
    """MongoDB populator with batch operations"""
    
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
    
    async def connect(self):
        """Connect to MongoDB"""
        print(f"\n🔌 Connecting to MongoDB...")
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # Test connection
        await self.client.admin.command('ping')
        print(f"✅ Connected to database: {self.db_name}.{self.collection_name}")
    
    async def create_indexes(self):
        """Create indexes for efficient querying"""
        print("\n🔧 Creating indexes...")
        
        # Index on track_id (primary identifier)
        await self.collection.create_index("track_id", unique=True)
        print("   ✅ track_id (unique)")
        
        # Index on track_name + artist_name (for search)
        await self.collection.create_index([("track_name", 1), ("artist_name", 1)])
        print("   ✅ track_name + artist_name")
        
        # Index on moods (for filtering)
        await self.collection.create_index("moods.primary_mood")
        await self.collection.create_index("moods.all_moods")
        print("   ✅ mood indexes")
        
        # Index on audio features (for similarity search)
        await self.collection.create_index("features.valence")
        await self.collection.create_index("features.energy")
        print("   ✅ feature indexes")
        
        print("✅ Indexes created")
    
    async def get_existing_track_ids(self) -> set:
        """Get set of existing track IDs to avoid duplicates"""
        print("\n📊 Checking existing tracks...")
        existing_ids = set()
        
        cursor = self.collection.find({}, {"track_id": 1})
        async for doc in cursor:
            existing_ids.add(doc["track_id"])
        
        print(f"✅ Found {len(existing_ids):,} existing tracks")
        return existing_ids
    
    async def bulk_insert(self, documents: List[Dict], ordered: bool = False):
        """Bulk insert documents with upsert"""
        if not documents:
            return 0
        
        operations = []
        for doc in documents:
            operations.append(
                UpdateOne(
                    {"track_id": doc["track_id"]},
                    {"$set": doc},
                    upsert=True
                )
            )
        
        try:
            result = await self.collection.bulk_write(operations, ordered=ordered)
            return result.upserted_count + result.modified_count
        except Exception as e:
            print(f"⚠️  Bulk insert error: {e}")
            return 0
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("\n✅ Database connection closed")


class GracefulExit:
    def __init__(self):
        self.cancelled = False
        try:
            signal.signal(signal.SIGINT, self._on_cancel)
            signal.signal(signal.SIGTERM, self._on_cancel)
        except Exception:
            pass

    def _on_cancel(self, *_):
        print("\n🛑 Graceful stop requested — finishing current batch...")
        self.cancelled = True


# ============================================
# MAIN PROCESSING PIPELINE
# ============================================

async def process_and_populate(
    csv_files: List[str],
    batch_size: int = 500,
    limit: Optional[int] = None,
    skip_existing: bool = True
):
    """Main pipeline: Load CSV -> Predict moods -> Populate MongoDB"""
    
    print("\n" + "="*80)
    print("🎵 SPOTIFY DATASET TO MONGODB POPULATOR")
    print("="*80)
    
    # 1. Load ML Model
    print("\n📦 Step 1: Loading ML Model...")
    predictor = MoodPredictor(MODEL_PATH, METADATA_PATH)
    predictor.load()
    
    # 2. Load Dataset
    print("\n📂 Step 2: Loading Spotify Dataset...")
    df = load_spotify_dataset(csv_files)
    df = extract_required_features(df)
    
    if limit:
        print(f"⚠️  Limiting to first {limit:,} tracks for testing")
        df = df.head(limit)
    
    # 3. Connect to MongoDB
    print("\n🔌 Step 3: Connecting to MongoDB...")
    db_populator = DatabasePopulator(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)
    await db_populator.connect()
    await db_populator.create_indexes()
    
    # 4. Get existing tracks
    existing_ids = set()
    if skip_existing:
        existing_ids = await db_populator.get_existing_track_ids()
    
    # 5. Process tracks in batches
    print("\n🔮 Step 4: Processing tracks and predicting moods...")
    print(f"   Batch size: {batch_size}")
    print(f"   Total tracks: {len(df):,}")
    
    total_processed = 0
    total_inserted = 0
    total_skipped = 0
    batch_documents = []
    
    # Create progress bar
    pbar = tqdm(total=len(df), desc="Processing", unit="track")
    
    stopper = GracefulExit()
    
    for idx, row in df.iterrows():
        
        if stopper.cancelled:
            print("\n🛑 Stop signal detected — flushing remaining tracks before exit...")
            break
        
        try:
            # Extract track metadata
            track_id = str(row.get('id', f"track_{idx}"))
            track_name = row.get('name') or row.get('track_name', 'Unknown Track')
            artist_name = row.get('artists') or row.get('artist') or row.get('artist_name', 'Unknown Artist')
            
            # Skip if already exists
            if skip_existing and track_id in existing_ids:
                total_skipped += 1
                pbar.update(1)
                continue
            
            # Extract audio features
            features = {}
            for feature in REQUIRED_FEATURES:
                value = row.get(feature)
                if pd.notna(value):
                    features[feature] = float(value)
                else:
                    features[feature] = 0.5  # Default value
            
            # Predict moods
            mood_prediction = predictor.predict(features)
            
            # Build document
            document = {
                "track_id": track_id,
                "track_name": track_name,
                "artist_name": artist_name,
                "features": features,
                "moods": {
                    "base_mood": mood_prediction['base_mood'],
                    "base_confidence": mood_prediction['base_confidence'],
                    "primary_mood": mood_prediction['primary_mood'],
                    "all_moods": mood_prediction['all_moods'],
                    "mood_scores": mood_prediction['mood_scores'],
                    "num_tags": mood_prediction['num_tags']
                },
                "metadata": {
                    "year": int(row.get('year', 0)) if pd.notna(row.get('year')) else None,
                    "popularity": int(row.get('popularity', 0)) if pd.notna(row.get('popularity')) else None,
                    "duration_ms": int(row.get('duration_ms', 0)) if pd.notna(row.get('duration_ms')) else None,
                    "release_date": str(row.get('release_date')) if pd.notna(row.get('release_date')) else None,
                },
                "created_at": datetime.utcnow(),
                "source": "spotify_kaggle_dataset"
            }
            
            batch_documents.append(document)
            total_processed += 1
            
            # Insert batch when full
            if len(batch_documents) >= batch_size:
                inserted = await db_populator.bulk_insert(batch_documents)
                total_inserted += inserted
                batch_documents = []
                pbar.set_postfix({
                    'inserted': total_inserted,
                    'skipped': total_skipped
                })
            
            pbar.update(1)
            
        except Exception as e:
            print(f"\n⚠️  Error processing track {idx}: {e}")
            pbar.update(1)
            continue
    
    # Insert remaining batch
    if batch_documents:
        inserted = await db_populator.bulk_insert(batch_documents)
        total_inserted += inserted
    
    pbar.close()
    
    # 6. Summary
    print("\n" + "="*80)
    print("✅ DATABASE POPULATION COMPLETE")
    print("="*80)
    print(f"Total processed: {total_processed:,}")
    print(f"Total inserted/updated: {total_inserted:,}")
    print(f"Total skipped (existing): {total_skipped:,}")
    print(f"Database: {DATABASE_NAME}.{COLLECTION_NAME}")
    print("="*80)
    
    # 7. Show sample documents
    print("\n📄 Sample documents:")
    cursor = db_populator.collection.find().limit(3)
    async for doc in cursor:
        print(f"\n   Track: {doc['track_name']}")
        print(f"   Artist: {doc['artist_name']}")
        print(f"   Primary Mood: {doc['moods']['primary_mood']}")
        print(f"   All Moods: {', '.join(doc['moods']['all_moods'])}")
        print(f"   Mood Scores: {doc['moods']['mood_scores']}")
    
    # 8. Collection stats
    print("\n📊 Collection Statistics:")
    total_count = await db_populator.collection.count_documents({})
    print(f"   Total tracks: {total_count:,}")
    
    # Count by primary mood
    pipeline = [
        {"$group": {"_id": "$moods.primary_mood", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    print("\n   Mood distribution:")
    async for doc in db_populator.collection.aggregate(pipeline):
        print(f"      {doc['_id']}: {doc['count']:,}")
    
    await db_populator.close()


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with Spotify dataset and mood predictions"
    )
    parser.add_argument(
        '--csv2018',
        type=str,
        help='Path to 2018 Spotify CSV file'
    )
    parser.add_argument(
        '--csv2019',
        type=str,
        help='Path to 2019 Spotify CSV file'
    )
    parser.add_argument(
        '--csv',
        type=str,
        action='append',
        help='Path to CSV file(s) (can be used multiple times)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Batch size for MongoDB inserts (default: 500)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of tracks to process (for testing)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Do not skip existing tracks (update them)'
    )
    
    args = parser.parse_args()
    
    # Build list of CSV files
    csv_files = []
    if args.csv2018:
        csv_files.append(args.csv2018)
    if args.csv2019:
        csv_files.append(args.csv2019)
    if args.csv:
        csv_files.extend(args.csv)
    
    if not csv_files:
        print("❌ Error: Please provide at least one CSV file")
        print("\nUsage:")
        print("  python populate_database.py --csv2018 spotify_2018.csv --csv2019 spotify_2019.csv")
        print("  python populate_database.py --csv data.csv --batch-size 1000")
        sys.exit(1)
    
    # Validate files exist
    for csv_file in csv_files:
        if not Path(csv_file).exists():
            print(f"❌ Error: File not found: {csv_file}")
            sys.exit(1)
    
    # Validate MongoDB URI
    if not MONGO_URI:
        print("❌ Error: MONGO_URI not found in environment variables")
        print("   Please set MONGO_URI in your .env file")
        sys.exit(1)
    
    # Validate model exists
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Run async pipeline
    asyncio.run(process_and_populate(
        csv_files=csv_files,
        batch_size=args.batch_size,
        limit=args.limit,
        skip_existing=not args.no_skip_existing
    ))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")