#!/usr/bin/env python3
"""
Populate MoodiQ song corpus into MongoDB (YTMusic-only).
Stores: 12 audio features + 1-3 mood tags per track
"""

import os
import sys
import math
import asyncio
import signal
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ytmusicapi import YTMusic
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import services
from services import music_service, model_service

# Configuration
MONGODB_URI = "mongodb+srv://nodemailer274:Ganesh%401971@cluster0.fgpremh.mongodb.net/moodiq?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "moodiq"
COLLECTION = "moodiq_songs"

TARGET_TOTAL = 100_000
DB_BATCH_SIZE = 500
TRACK_CONCURRENCY = 16

# Feature order (12)
FEATURE_ORDER = [
    "valence", "energy", "danceability", "acousticness", "instrumentalness",
    "speechiness", "tempo", "loudness", "liveness", "key", "mode", "time_signature"
]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("populate")

# Graceful exit handler
class GracefulExit:
    def __init__(self):
        self.cancelled = False
        try:
            signal.signal(signal.SIGINT, self._on_cancel)
            signal.signal(signal.SIGTERM, self._on_cancel)
        except Exception:
            pass

    def _on_cancel(self, *_):
        self.cancelled = True


# YTMusic client
ytmusic_client: Optional[YTMusic] = None

def init_ytmusic_once():
    global ytmusic_client
    if ytmusic_client is None:
        try:
            ytmusic_client = YTMusic()
            print("✅ YTMusic initialized")
        except Exception as e:
            print(f"❌ YTMusic init failed: {e}")
            ytmusic_client = None


YTMUSIC_COUNTRIES = [
    "US","GB","IN","CA","AU","DE","FR","BR","MX","ES","IT","NL","SE","NO","DK",
    "IE","PL","PT","TR","IL","AE","SA","ZA","EG","NG","KE","JP","KR","TW",
    "HK","SG","TH","MY","ID","PH","VN","AR","CL","CO","PE","UY","EC","CR",
]


def _dedupe_key(name: Optional[str], artists: Optional[List[str]]) -> Optional[Tuple[str,str]]:
    if not name or not artists:
        return None
    artist = artists[0] if artists else ""
    return (name.strip().lower(), artist.strip().lower())


def _yt_song_to_minimal(song: Dict) -> Optional[Dict]:
    try:
        title = song.get("title")
        artists = [a["name"] for a in song.get("artists", []) if a.get("name")]
        album = song.get("album", {}).get("name") if song.get("album") else None
        duration_sec = song.get("duration_seconds") or song.get("lengthSeconds")
        if isinstance(duration_sec, str):
            try:
                duration_sec = int(duration_sec)
            except:
                duration_sec = None
        video_id = song.get("videoId") or song.get("video_id")

        if not title or not artists:
            return None

        return {
            "id": video_id,
            "name": title,
            "artist": artists[0],
            "artists": artists,
            "album": album,
            "duration_ms": (duration_sec or 0) * 1000,
            "source": "youtube_music"
        }
    except Exception:
        return None


async def harvest_from_charts(max_per_country: int = 200) -> List[Dict]:
    init_ytmusic_once()
    if not ytmusic_client:
        return []

    seen: Set[Tuple[str,str]] = set()
    out: List[Dict] = []

    for cc in YTMUSIC_COUNTRIES:
        try:
            charts = ytmusic_client.get_charts(country=cc)
            song_block = charts.get("songs", {})
            items = song_block.get("items", [])[:max_per_country]
            count_before = len(out)
            for it in items:
                m = _yt_song_to_minimal(it)
                if not m:
                    continue
                k = _dedupe_key(m["name"], m["artists"])
                if not k or k in seen:
                    continue
                seen.add(k)
                out.append(m)
            print(f"🌍 Charts {cc}: +{len(out)-count_before} (total {len(out)})")
            await asyncio.sleep(0.15)
        except Exception as e:
            print(f"⚠️ charts({cc}) failed: {e}")

    return out


def _make_search_seeds(target: int) -> List[str]:
    letters = [chr(c) for c in range(ord('a'), ord('z')+1)]
    digits = [str(d) for d in range(10)]
    keywords = [
        "official audio", "official video", "lyrics", "remaster", "live",
        "pop", "rock", "hip hop", "rap", "edm", "electronic", "house",
        "indie", "alternative", "r&b", "soul", "jazz", "blues",
        "latin", "k-pop", "j-pop", "bollywood", "punjabi", "hindi", "tamil", "telugu"
    ]
    seeds = letters + digits + keywords

    if target > 50_000:
        bigrams = [a+b for a in letters for b in letters]
        seeds += bigrams[:400]
    return seeds


async def harvest_from_search(target_count: int = 100_000, per_seed: int = 250) -> List[Dict]:
    init_ytmusic_once()
    if not ytmusic_client:
        return []

    seeds = _make_search_seeds(target_count)
    seen: Set[Tuple[str,str]] = set()
    out: List[Dict] = []

    for i, seed in enumerate(seeds, 1):
        if len(out) >= target_count:
            break
        try:
            pulls = math.ceil(per_seed / 100)
            pulled_this_seed = 0
            for _ in range(pulls):
                if len(out) >= target_count:
                    break
                res = ytmusic_client.search(seed, filter="songs", limit=100)
                added = 0
                for it in res:
                    m = _yt_song_to_minimal(it)
                    if not m:
                        continue
                    k = _dedupe_key(m["name"], m["artists"])
                    if not k or k in seen:
                        continue
                    seen.add(k)
                    out.append(m)
                    added += 1
                pulled_this_seed += added
                await asyncio.sleep(0.2)
            print(f"🔎 Seed [{i}/{len(seeds)}] '{seed}': +{pulled_this_seed} (total {len(out)})")
        except Exception as e:
            print(f"⚠️ search('{seed}') failed: {e}")

    return out[:target_count]


async def yield_tracks_ytmusic_only(
    target_total: int = 100_000,
    charts_per_country: int = 200,
    search_per_seed: int = 250
):
    charts = await harvest_from_charts(max_per_country=charts_per_country)
    seen: Set[Tuple[str,str]] = set()
    emitted = 0

    for m in charts:
        k = _dedupe_key(m["name"], m["artists"])
        if not k or k in seen:
            continue
        seen.add(k)
        yield m
        emitted += 1
        if emitted >= target_total:
            return

    search_bulk = await harvest_from_search(
        target_count=max(0, target_total - emitted),
        per_seed=search_per_seed
    )
    for m in search_bulk:
        k = _dedupe_key(m["name"], m["artists"])
        if not k or k in seen:
            continue
        seen.add(k)
        yield m
        emitted += 1
        if emitted >= target_total:
            return


async def ensure_indexes(coll):
    await coll.create_index([("name", 1), ("artist", 1)], unique=True)


async def bulk_upsert_batch(coll, docs: List[Dict[str, Any]]):
    if not docs:
        return

    from pymongo import UpdateOne
    ops = []
    for d in docs:
        key = {"name": d["name"], "artist": d["artist"]}
        ops.append(UpdateOne(key, {"$set": d}, upsert=True))

    try:
        res = await coll.bulk_write(ops, ordered=False)
        inserted = (res.upserted_count or 0) + (res.modified_count or 0)
        log.info(f"🗃️  Mongo upsert batch: {len(docs)} attempted, {inserted} upserted/modified")
    except Exception as e:
        log.warning(f"Mongo bulk_write had errors: {e}")


async def process_track(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        name = candidate.get("name")
        artist = candidate.get("artist")
        if not name or not artist:
            return None

        # Get 12 audio features
        feats = await music_service.get_audio_features(name, artist)
        if not feats:
            return None

        # Get mood tags (1-3 tags)
        lyrics_sentiment = {"polarity": 0.0, "subjectivity": 0.0}
        mood_data = await model_service.predict_mood_from_features(
            feats,
            lyrics_sentiment,
            user_id=None,
            track_id=None,
            genre=None
        )

        # Extract 1-3 mood tags
        all_moods = mood_data.get('all_moods', [mood_data.get('primary_mood', 'Relaxed')])
        mood_tags = all_moods[:3]  # Max 3 tags
        
        if len(mood_tags) == 0:
            mood_tags = [mood_data.get('primary_mood', 'Relaxed')]

        # Build document with 12 features + 1-3 mood tags
        doc = {
            "name": name,
            "artist": artist,
            "mood_tags": mood_tags,
            "features": {k: feats.get(k) for k in FEATURE_ORDER},
            "source": candidate.get("source", "youtube_music"),
            "yt_video_id": candidate.get("id"),
            "created_at": datetime.utcnow(),
        }
        return doc

    except Exception as e:
        log.debug(f"process_track error: {e}")
        return None


async def main():
    stopper = GracefulExit()

    # MongoDB connection
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION]
    await ensure_indexes(coll)
    log.info(f"✅ Mongo connected: db={DB_NAME}, coll={COLLECTION}")

    # Load model service (already initialized in services)
    model_service.load_model()
    log.info("✅ Model service initialized")

    target_total = TARGET_TOTAL
    charts_per_country = 0
    search_per_seed = 250

    buffer_docs: List[Dict[str, Any]] = []
    sem_tracks = asyncio.Semaphore(TRACK_CONCURRENCY)

    async def process_with_limit(t):
        async with sem_tracks:
            return await process_track(t)

    count_seen = 0
    async for track_min in yield_tracks_ytmusic_only(
        target_total=target_total,
        charts_per_country=charts_per_country,
        search_per_seed=search_per_seed
    ):
        if stopper.cancelled:
            break

        doc = await process_with_limit(track_min)
        if doc is None:
            continue

        buffer_docs.append(doc)
        count_seen += 1

        if len(buffer_docs) >= DB_BATCH_SIZE:
            await bulk_upsert_batch(coll, buffer_docs)
            log.info(f"💾 Inserted batch of {len(buffer_docs)} (total processed: {count_seen}/{target_total})")
            buffer_docs.clear()

    # Final flush
    if buffer_docs:
        await bulk_upsert_batch(coll, buffer_docs)
        buffer_docs.clear()

    log.info("✅ Completed corpus population.")
    client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
