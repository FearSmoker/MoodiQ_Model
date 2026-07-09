"""
Live Spotify Catalog -> MongoDB Populator (2026)
=================================================
Replaces the old populate_database.py, which ingested from spotify_2018.csv /
spotify_2019.csv. Those datasets are frozen in time and were never going to
contain anything released after 2019 — not what you want for a "2026 catalog"
recommendation engine.

WHY THIS LOOKS THE WAY IT DOES (read before changing endpoints):
- GET /v1/recommendations and GET /v1/audio-features are dead for this app
  (deprecated Nov 2024, and Spotify tightened further in Feb 2026). There is
  no personal-Premium override for this — Premium only keeps a Development
  Mode app from being shut off (Feb 2026 policy), it does not grant Extended
  Quota Mode. Real audio features for freshly-discovered tracks come from
  services.music_service.get_audio_features(), which already implements the
  MusicBrainz -> AcousticBrainz -> Gemini fallback chain. We call it WITHOUT
  an access_token so it skips the (dead) Spotify features attempt entirely
  and goes straight to that pipeline.
- GET /search is alive but capped at 10 results/request as of Feb 2026 — you
  must paginate via `offset` to get real volume. We do that.
- Batch endpoints (GET /artists, GET /albums, GET /tracks plural) were fully
  removed in Feb 2026 — fetch one artist/album at a time.
- GET /artists/{id}/albums and GET /albums/{id}/tracks are still alive and
  are how we reach tracks that a generic genre search won't surface (an
  artist's latest album, for instance) — this is the main way we bias
  toward *current* catalog rather than whatever `/search` happens to rank
  highest.
- GET /artists/{id}/top-tracks and GET /browse/new-releases were REMOVED in
  Feb 2026 — do not use them, they will 404/403.

Requirements:
    pip install motor pymongo onnxruntime numpy tqdm python-dotenv spotipy

Usage:
    python populate_database.py --target-count 3000 --batch-size 200
    python populate_database.py --genres "pop,indie,hip hop" --since-year 2024
"""

import os
import sys
import asyncio
import argparse
import random
import signal
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

import numpy as np
import onnxruntime as ort
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sys.path.insert(0, str(Path(__file__).resolve().parent))
from services import music_service  # noqa: E402  (MusicBrainz -> AcousticBrainz -> Gemini pipeline)

load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "moodiq"
COLLECTION_NAME = "databasesongs"

MODEL_PATH = "models/mood_model.onnx"
METADATA_PATH = "models/model_metadata.json"

# The 12 features the ONNX model + mood-similarity scoring expect.
REQUIRED_FEATURES = [
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

BASE_MOOD_CLASSES = ["Calm", "Energetic", "Happy", "Sad"]

EXTENDED_MOODS = {
    "Relaxed":     {"base_moods": ["Calm"], "profile": {"valence": 0.5, "energy": 0.25, "danceability": 0.3},
                     "weights": {"valence": 0.8, "energy": 1.0, "acousticness": 0.9}},
    "Focused":     {"base_moods": ["Calm"], "profile": {"valence": 0.45, "energy": 0.4, "danceability": 0.35},
                     "weights": {"energy": 1.0, "speechiness": 0.9, "instrumentalness": 0.8}},
    "Romantic":    {"base_moods": ["Calm", "Happy"], "profile": {"valence": 0.6, "energy": 0.35, "danceability": 0.4},
                     "weights": {"valence": 0.9, "energy": 0.8, "acousticness": 0.7}},
    "Excited":     {"base_moods": ["Happy", "Energetic"], "profile": {"valence": 0.8, "energy": 0.85, "danceability": 0.8},
                     "weights": {"valence": 0.9, "energy": 1.0, "danceability": 0.9}},
    "Angry":       {"base_moods": ["Energetic", "Sad"], "profile": {"valence": 0.25, "energy": 0.85, "danceability": 0.5},
                     "weights": {"valence": 1.0, "energy": 1.0, "loudness": 0.9}},
    "Chill":       {"base_moods": ["Calm", "Happy"], "profile": {"valence": 0.6, "energy": 0.3, "danceability": 0.45},
                     "weights": {"energy": 1.0, "valence": 0.8, "danceability": 0.7}},
    "Melancholic": {"base_moods": ["Sad"], "profile": {"valence": 0.2, "energy": 0.25, "danceability": 0.3},
                     "weights": {"valence": 1.0, "energy": 0.9, "acousticness": 0.8}},
    "Dreamy":      {"base_moods": ["Calm", "Sad"], "profile": {"valence": 0.4, "energy": 0.3, "danceability": 0.35},
                     "weights": {"energy": 0.9, "acousticness": 1.0, "instrumentalness": 0.8}},
    "Motivated":   {"base_moods": ["Energetic"], "profile": {"valence": 0.65, "energy": 0.75, "danceability": 0.65},
                     "weights": {"energy": 1.0, "valence": 0.8, "tempo": 0.7}},
    "Joyful":      {"base_moods": ["Happy"], "profile": {"valence": 0.85, "energy": 0.7, "danceability": 0.75},
                     "weights": {"valence": 1.0, "energy": 0.8, "danceability": 0.9}},
    "Ambient":     {"base_moods": ["Calm"], "profile": {"valence": 0.5, "energy": 0.2, "danceability": 0.25},
                     "weights": {"instrumentalness": 1.0, "energy": 0.9, "speechiness": 0.8}},
    "Party":       {"base_moods": ["Energetic", "Happy"], "profile": {"valence": 0.8, "energy": 0.9, "danceability": 0.9},
                     "weights": {"danceability": 1.0, "energy": 1.0, "valence": 0.9}},
}
ALL_MOOD_LABELS = list(EXTENDED_MOODS.keys())

# Genre seeds used to steer discovery toward each mood bucket. Mirrors
# MOOD_GENRE_SEEDS in spotify_service.py so ingestion and live recommendation
# discovery pull from the same genre vocabulary.
MOOD_GENRE_SEEDS = {
    'Joyful': ['pop', 'dance pop'], 'Excited': ['edm', 'electropop'],
    'Party': ['party', 'hip hop'], 'Melancholic': ['sad', 'indie folk'],
    'Dreamy': ['dream pop', 'shoegaze'], 'Relaxed': ['acoustic', 'chill'],
    'Chill': ['chillhop', 'indie pop'], 'Focused': ['instrumental', 'ambient'],
    'Romantic': ['r&b', 'soul'], 'Motivated': ['workout', 'rock'],
    'Angry': ['metal', 'punk'], 'Ambient': ['ambient', 'atmospheric'],
}
DEFAULT_GENRES = sorted({g for genres in MOOD_GENRE_SEEDS.values() for g in genres})

# ============================================
# MOOD MODEL (unchanged from the CSV version)
# ============================================

class MoodPredictor:

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.session = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.mood_classes = BASE_MOOD_CLASSES

    def load(self):
        print(f"📦 Loading ONNX model from {self.model_path}...")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, session_options,
                                             providers=['CPUExecutionProvider'])
        print("✅ ONNX model loaded")

        if Path(self.metadata_path).exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.scaler_mean = np.array(metadata['scaler_mean'], dtype=np.float32)
                self.scaler_scale = np.array(metadata['scaler_scale'], dtype=np.float32)
                self.mood_classes = metadata.get('mood_classes', BASE_MOOD_CLASSES)
                print("✅ Metadata loaded")
        else:
            print("⚠️  Metadata not found, using unscaled features")

    def normalize_features(self, features: Dict) -> np.ndarray:
        feature_values = [float(features.get(name, 0.5)) for name in REQUIRED_FEATURES]
        arr = np.array(feature_values, dtype=np.float32)
        if self.scaler_mean is not None and self.scaler_scale is not None:
            arr = ((arr - self.scaler_mean) / self.scaler_scale).astype(np.float32)
        return arr

    def calculate_mood_similarity(self, features: Dict, mood_name: str) -> float:
        mood_profile = EXTENDED_MOODS[mood_name]["profile"]
        weights = EXTENDED_MOODS[mood_name]["weights"]
        total_similarity, total_weight = 0.0, 0.0

        for feature, target_value in mood_profile.items():
            if feature not in features:
                continue
            actual_value = features[feature]
            weight = weights.get(feature, 0.5)

            if feature == 'tempo':
                actual_n = (actual_value - 40) / 160
                target_n = (target_value - 40) / 160
            elif feature == 'loudness':
                actual_n = (actual_value + 60) / 60
                target_n = (target_value + 60) / 60
            else:
                actual_n, target_n = actual_value, target_value

            similarity = 1.0 - abs(actual_n - target_n)
            total_similarity += similarity * weight
            total_weight += weight

        return total_similarity / total_weight if total_weight > 0 else 0.0

    def get_multi_mood_tags(self, features: Dict, min_similarity: float = 0.70, max_tags: int = 3) -> List[tuple]:
        scores = [(m, self.calculate_mood_similarity(features, m)) for m in ALL_MOOD_LABELS]
        scores = [s for s in scores if s[1] >= min_similarity]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max_tags]

    def predict_base_mood(self, features: Dict) -> tuple:
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        input_data = self.normalize_features(features).reshape(1, -1).astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        probabilities = outputs[0][0]
        predicted_index = int(np.argmax(probabilities))
        return self.mood_classes[predicted_index], float(probabilities[predicted_index])

    def predict(self, features: Dict) -> Dict:
        base_mood, base_confidence = self.predict_base_mood(features)
        mood_tags = self.get_multi_mood_tags(features, min_similarity=0.70, max_tags=3)
        if not mood_tags:
            mood_tags = [(base_mood, base_confidence)]
        return {
            'base_mood': base_mood,
            'base_confidence': base_confidence,
            'primary_mood': mood_tags[0][0],
            'all_moods': [m for m, _ in mood_tags],
            'mood_scores': {m: float(s) for m, s in mood_tags},
            'num_tags': len(mood_tags),
        }

# ============================================
# LIVE CATALOG DISCOVERY (replaces CSV loading)
# ============================================

def get_client_credentials_client() -> spotipy.Spotify:
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in environment")
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth_manager, requests_timeout=15, retries=3)

class LiveCatalogDiscovery:

    def __init__(self, sp: spotipy.Spotify, since_year: int = 2023):
        self.sp = sp
        self.since_year = since_year
        self.seen_ids: Set[str] = set()

    def _search_page(self, query: str, offset: int, limit: int = 10, type_: str = 'track'):
        try:
            return self.sp.search(q=query, type=type_, limit=limit, offset=offset)
        except Exception as e:
            print(f"⚠️ Search failed for '{query}' @offset={offset}: {e}")
            return None

    def discover_via_search(self, genres: List[str], target_count: int, max_offset: int = 190) -> List[Dict]:
        
        results = []
        year_filter = f"year:{self.since_year}-2026"

        for genre in genres:
            if len(results) >= target_count:
                break
            query = f'genre:"{genre}" {year_filter}'
            offset = 0
            while offset <= max_offset and len(results) < target_count:
                page = self._search_page(query, offset=offset, limit=10)
                offset += 10
                if not page:
                    break
                items = page.get('tracks', {}).get('items', [])
                if not items:
                    break
                for track in items:
                    if not track or not track.get('id') or track['id'] in self.seen_ids:
                        continue
                    self.seen_ids.add(track['id'])
                    results.append(self._track_to_record(track, source_genre=genre))
        print(f"🔍 Search discovery: {len(results)} tracks (genres={genres}, since={self.since_year})")
        return results

    def discover_via_artist_albums(self, seed_artist_ids: List[str], target_count: int) -> List[Dict]:
        
        results = []
        for artist_id in seed_artist_ids:
            if len(results) >= target_count:
                break
            try:
                albums_resp = self.sp.artist_albums(
                    artist_id, album_type='album,single', limit=20
                )
            except Exception as e:
                print(f"⚠️ artist_albums failed for {artist_id}: {e}")
                continue

            albums = albums_resp.get('items', []) if albums_resp else []
            # Prefer recent releases first
            def _release_year(a):
                d = a.get('release_date') or '0000'
                try:
                    return int(d[:4])
                except Exception:
                    return 0
            albums.sort(key=_release_year, reverse=True)

            for album in albums:
                if len(results) >= target_count:
                    break
                if _release_year(album) < self.since_year:
                    continue  # skip older back-catalog albums entirely
                album_id = album.get('id')
                if not album_id:
                    continue
                try:
                    tracks_resp = self.sp.album_tracks(album_id, limit=50)
                except Exception as e:
                    print(f"⚠️ album_tracks failed for {album_id}: {e}")
                    continue

                for track in (tracks_resp.get('items', []) if tracks_resp else []):
                    if not track or not track.get('id') or track['id'] in self.seen_ids:
                        continue
                    self.seen_ids.add(track['id'])
                    # album_tracks items don't include the parent album object;
                    # attach what we already know from `album`.
                    track = dict(track)
                    track['album'] = album
                    results.append(self._track_to_record(track, source_genre=None))
                    if len(results) >= target_count:
                        break

        print(f"🎼 Artist/album crawl: {len(results)} additional tracks")
        return results

    @staticmethod
    def _track_to_record(track: Dict, source_genre: Optional[str]) -> Dict:
        artists = track.get('artists') or []
        album = track.get('album') or {}
        return {
            'spotify_id': track.get('id'),
            'track_name': track.get('name'),
            'artist_name': artists[0]['name'] if artists else 'Unknown Artist',
            'artist_id': artists[0]['id'] if artists else None,
            'album_name': album.get('name'),
            'release_date': album.get('release_date'),
            'source_genre': source_genre,
        }

# ============================================
# DATABASE OPERATIONS (unchanged)
# ============================================

class DatabasePopulator:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

    async def connect(self):
        print(f"\n🔌 Connecting to MongoDB...")
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        await self.client.admin.command('ping')
        print(f"✅ Connected to database: {self.db_name}.{self.collection_name}")

    async def create_indexes(self):
        print("\n🔧 Creating indexes...")
        await self.collection.create_index("track_id", unique=True)
        await self.collection.create_index([("track_name", 1), ("artist_name", 1)])
        await self.collection.create_index("moods.primary_mood")
        await self.collection.create_index("moods.all_moods")
        await self.collection.create_index("features.valence")
        await self.collection.create_index("features.energy")
        print("✅ Indexes created")

    async def get_existing_track_ids(self) -> set:
        print("\n📊 Checking existing tracks...")
        existing_ids = set()
        cursor = self.collection.find({}, {"track_id": 1})
        async for doc in cursor:
            existing_ids.add(doc["track_id"])
        print(f"✅ Found {len(existing_ids):,} existing tracks")
        return existing_ids

    async def bulk_insert(self, documents: List[Dict], ordered: bool = False):
        if not documents:
            return 0
        operations = [
            UpdateOne({"track_id": doc["track_id"]}, {"$set": doc}, upsert=True)
            for doc in documents
        ]
        try:
            result = await self.collection.bulk_write(operations, ordered=ordered)
            return result.upserted_count + result.modified_count
        except Exception as e:
            print(f"⚠️  Bulk insert error: {e}")
            return 0

    async def close(self):
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
# MAIN PIPELINE
# ============================================

def _fill_feature_defaults(features: Dict) -> Dict:
    
    features = dict(features)
    features.setdefault('mode', 1)
    features.setdefault('time_signature', 4)
    for name in REQUIRED_FEATURES:
        if name not in features or features[name] is None:
            features[name] = 0.5
    return features

async def process_and_populate(
    target_count: int,
    genres: List[str],
    since_year: int,
    batch_size: int = 200,
    concurrency: int = 8,
    skip_existing: bool = True,
):
    print("\n" + "=" * 80)
    print("🎵 LIVE SPOTIFY CATALOG -> MONGODB POPULATOR (2026)")
    print("=" * 80)

    if not MONGO_URI:
        print("❌ Error: MONGO_URI not set in environment")
        sys.exit(1)
    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("\n📦 Step 1: Loading ML Model...")
    predictor = MoodPredictor(MODEL_PATH, METADATA_PATH)
    predictor.load()

    print(f"\n📡 Step 2: Discovering live catalog (target={target_count:,}, since={since_year})...")
    sp = get_client_credentials_client()
    discovery = LiveCatalogDiscovery(sp, since_year=since_year)

    search_results = discovery.discover_via_search(genres, target_count=int(target_count * 0.6))
    seed_artist_ids = list({r['artist_id'] for r in search_results if r.get('artist_id')})
    random.shuffle(seed_artist_ids)
    crawl_results = discovery.discover_via_artist_albums(
        seed_artist_ids[:150], target_count=target_count - len(search_results)
    )

    catalog = search_results + crawl_results
    if not catalog:
        print("❌ No tracks discovered — check SPOTIFY_CLIENT_ID/SECRET and network access")
        sys.exit(1)
    print(f"✅ Total discovered: {len(catalog):,} tracks")

    print("\n🔌 Step 3: Connecting to MongoDB...")
    db_populator = DatabasePopulator(MONGO_URI, DATABASE_NAME, COLLECTION_NAME)
    await db_populator.connect()
    await db_populator.create_indexes()

    existing_ids = await db_populator.get_existing_track_ids() if skip_existing else set()

    #    and run mood prediction, with bounded concurrency since this hits
    #    three external services per track.
    print("\n🔮 Step 4: Fetching real audio features + predicting moods...")
    print(f"   Concurrency: {concurrency}, Batch size: {batch_size}")

    sem = asyncio.Semaphore(concurrency)
    stopper = GracefulExit()
    total_inserted = 0
    total_skipped = 0
    total_failed = 0
    batch_documents: List[Dict] = []

    async def _process_one(record: Dict) -> Optional[Dict]:
        track_id = record['spotify_id']
        if skip_existing and track_id in existing_ids:
            return None
        async with sem:
            try:
                # No access_token passed on purpose: Spotify's own
                # audio-features endpoint is dead for this app, so skip
                # straight to MusicBrainz -> AcousticBrainz -> Gemini.
                raw_features = await music_service.get_audio_features(
                    record['track_name'],
                    record['artist_name'],
                    genre=record.get('source_genre'),
                    use_gemini_fallback=True,
                    access_token=None,
                )
                features = _fill_feature_defaults(raw_features)
                mood_prediction = predictor.predict(features)

                release_year = None
                if record.get('release_date'):
                    try:
                        release_year = int(record['release_date'][:4])
                    except Exception:
                        pass

                return {
                    "track_id": track_id,
                    "track_name": record['track_name'],
                    "artist_name": record['artist_name'],
                    "features": features,
                    "moods": {
                        "base_mood": mood_prediction['base_mood'],
                        "base_confidence": mood_prediction['base_confidence'],
                        "primary_mood": mood_prediction['primary_mood'],
                        "all_moods": mood_prediction['all_moods'],
                        "mood_scores": mood_prediction['mood_scores'],
                        "num_tags": mood_prediction['num_tags'],
                    },
                    "metadata": {
                        "album": record.get('album_name'),
                        "release_date": record.get('release_date'),
                        "year": release_year,
                        # popularity is no longer returned by Spotify as of
                        # Feb 2026 — intentionally omitted rather than faked.
                    },
                    "created_at": datetime.utcnow(),
                    "source": "spotify_live_catalog_2026",
                    "feature_source": features.get("source", "unknown"),
                }
            except Exception as e:
                print(f"\n⚠️ Failed processing '{record['track_name']}' by "
                      f"'{record['artist_name']}': {e}")
                return None

    pbar = tqdm(total=len(catalog), desc="Processing", unit="track")

    for i in range(0, len(catalog), batch_size):
        if stopper.cancelled:
            print("\n🛑 Stop signal detected — flushing remaining tracks before exit...")
            break

        chunk = catalog[i:i + batch_size]
        docs = await asyncio.gather(*[_process_one(r) for r in chunk])

        for doc in docs:
            pbar.update(1)
            if doc is None:
                total_skipped += 1
                continue
            batch_documents.append(doc)

        if batch_documents:
            inserted = await db_populator.bulk_insert(batch_documents)
            total_inserted += inserted
            batch_documents = []
            pbar.set_postfix({'inserted': total_inserted, 'skipped': total_skipped})

    if batch_documents:
        total_inserted += await db_populator.bulk_insert(batch_documents)

    pbar.close()

    # 5. Summary
    print("\n" + "=" * 80)
    print("✅ DATABASE POPULATION COMPLETE")
    print("=" * 80)
    print(f"Discovered: {len(catalog):,}")
    print(f"Inserted/updated: {total_inserted:,}")
    print(f"Skipped (existing or failed): {total_skipped:,}")
    print(f"Database: {DATABASE_NAME}.{COLLECTION_NAME}")
    print("=" * 80)

    print("\n📊 Collection Statistics:")
    total_count = await db_populator.collection.count_documents({})
    print(f"   Total tracks: {total_count:,}")

    pipeline = [
        {"$group": {"_id": "$moods.primary_mood", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
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
        description="Populate MongoDB from the LIVE Spotify catalog (no CSVs)."
    )
    parser.add_argument('--target-count', type=int, default=3000,
                         help='Approximate number of tracks to discover and ingest (default: 3000)')
    parser.add_argument('--genres', type=str, default=None,
                         help='Comma-separated genres to search (default: all mood-mapped genres)')
    parser.add_argument('--since-year', type=int, default=2023,
                         help='Only ingest tracks released on/after this year (default: 2023)')
    parser.add_argument('--batch-size', type=int, default=200,
                         help='Mongo upsert batch size (default: 200)')
    parser.add_argument('--concurrency', type=int, default=8,
                         help='Concurrent feature-fetch requests (default: 8 — AcousticBrainz/Gemini rate limits apply)')
    parser.add_argument('--no-skip-existing', action='store_true',
                         help='Re-fetch and update tracks already in the collection')

    args = parser.parse_args()
    genres = [g.strip() for g in args.genres.split(',')] if args.genres else DEFAULT_GENRES

    asyncio.run(process_and_populate(
        target_count=args.target_count,
        genres=genres,
        since_year=args.since_year,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        skip_existing=not args.no_skip_existing,
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")