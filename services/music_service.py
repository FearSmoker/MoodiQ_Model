"""
music_service.py

Implements:
- Single-track audio feature retrieval (MusicBrainz -> AcousticBrainz -> Gemini)
- Robust playlist aggregation (12 audio features)
- Playlist mood prediction (uses model_service)
- Flow optimizer (calm -> energetic ordering + smoothing)
- Recommendations via MongoDB vector search (top-N)
- Batch helpers and testing harness

Keep existing APIs (YTMusic, Last.fm, MusicBrainz, AcousticBrainz, Gemini).
"""

import os
import math
import requests
import asyncio
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import musicbrainzngs
from ytmusicapi import YTMusic
from motor.motor_asyncio import AsyncIOMotorClient

from . import cache_service, gemini_service, model_service

# -------------------------
# Configuration & Clients
# -------------------------
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0"
ACOUSTICBRAINZ_BASE_URL = "https://acousticbrainz.org/api/v1"

LASTFM_API_KEY = None
LASTFM_API_SECRET = None
_lastfm_initialized = False

# YTMusic client (best-effort init)
ytmusic = None
def init_ytmusic():
    global ytmusic
    try:
        ytmusic = YTMusic()
        print("✅ YTMusic initialized")
    except Exception as e:
        print(f"⚠️ YTMusic init failed: {e}")
        ytmusic = None
init_ytmusic()

# MusicBrainz useragent
musicbrainzngs.set_useragent("MoodiQ", "1.0", "https://moodiq.netlify.app")

# MongoDB (lazy connect)
_mongo_client: Optional[AsyncIOMotorClient] = None
def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise RuntimeError("MONGO_URI not set")
        _mongo_client = AsyncIOMotorClient(mongo_uri)
    return _mongo_client

# Features expected across the system (consistent naming)
FEATURE_KEYS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "spec_rate", "key", "duration_ms"
]
# Note: key & duration_ms are kept as integer/numeric metadata; model uses first 10 normalized features.
NUM_MODEL_FEATURES = 10  # model expects 10 features (as in your model_service)


# -------------------------
# Last.fm lazy init
# -------------------------
def _init_lastfm():
    global LASTFM_API_KEY, LASTFM_API_SECRET, _lastfm_initialized
    if _lastfm_initialized:
        return
    _lastfm_initialized = True
    LASTFM_API_KEY = os.getenv("LASTFM_API_KEY") or os.getenv("API_KEY_LASTFM")
    LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET") or os.getenv("API_SECRET_LASTFM")
    if LASTFM_API_KEY:
        print("✅ Last.fm API configured")
    else:
        print("⚠️ Last.fm key not found - tag lookups will be limited")

async def get_lastfm_tags(track_name: str, artist_name: str) -> List[str]:
    """Fetch top tags/genres for a song from Last.fm"""
    _init_lastfm()
    api_key = os.getenv("API_KEY_LASTFM") or os.getenv("LASTFM_API_KEY")
    if not api_key:
        return []
        
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.gettoptags",
        "artist": artist_name,
        "track": track_name,
        "api_key": api_key,
        "format": "json"
    }
    
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                toptags = data.get("toptags", {}).get("tag", [])
                if isinstance(toptags, list):
                    return [tag.get("name") for tag in toptags if tag.get("name")][:5]
                elif isinstance(toptags, dict):
                    return [toptags.get("name")] if toptags.get("name") else []
    except Exception as e:
        print(f"⚠️ Last.fm API error: {e}")
        
    return []



# -------------------------
# Helper utilities
# -------------------------
def _safe_mean(values: List[float]) -> float:
    """Compute mean safely; return 0.0 if empty."""
    if not values:
        return 0.0
    return float(np.mean(values))

def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def _normalize_spec_rate_from_tempo(tempo: float) -> float:
    # tempo expected around 40-200 -> spec_rate normalized 0-1 using same mapping
    try:
        tempo = float(tempo)
        return _clip((tempo - 40.0) / 160.0, 0.0, 1.0)
    except Exception:
        return 0.5

def _ensure_spec_rate(features: Dict[str, Any]) -> None:
    if "spec_rate" not in features or features.get("spec_rate") is None:
        tempo = features.get("tempo", 120.0)
        features["spec_rate"] = _normalize_spec_rate_from_tempo(tempo)


# -------------------------
# YTMusic search (preserve)
# -------------------------
async def search_tracks(query: str, limit: int = 20) -> List[Dict]:
    cache_key = f"ytmusic:search:{query}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached

    if not ytmusic:
        print("⚠️ YTMusic not initialized")
        return []

    try:
        results = ytmusic.search(query, filter="songs", limit=limit)
        tracks = []
        for item in results:
            tracks.append({
                "id": item.get("videoId"),
                "name": item.get("title"),
                "artists": [a["name"] for a in item.get("artists", [])],
                "album": item.get("album", {}).get("name") if item.get("album") else None,
                "duration_ms": (item.get("duration_seconds") or 0) * 1000,
                "thumbnail": (item.get("thumbnails") or [{}])[0].get("url"),
                "source": "youtube_music"
            })
        await cache_service.set_in_cache(cache_key, tracks, expiration=3600)
        return tracks
    except Exception as e:
        print(f"❌ YTMusic search error: {e}")
        return []


async def get_track_info(track_id: Optional[str] = None, track_name: Optional[str] = None, artist_name: Optional[str] = None) -> Optional[Dict]:
    cache_key = f"ytmusic:track:{track_id or (track_name + ':' + (artist_name or ''))}"
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached
    try:
        if track_id and ytmusic:
            song = ytmusic.get_song(track_id)
            info = {
                "id": song.get("videoId"),
                "name": song.get("title"),
                "artists": [a["name"] for a in song.get("artists", [])],
                "album": song.get("album", {}).get("name") if song.get("album") else None,
                "duration_ms": int(song.get("lengthSeconds", 0)) * 1000,
                "thumbnail": (song.get("thumbnails") or [{}])[-1].get("url"),
                "source": "youtube_music"
            }
        elif track_name and artist_name:
            results = await search_tracks(f"{track_name} {artist_name}", limit=1)
            if not results:
                return None
            info = results[0]
        else:
            return None
        await cache_service.set_in_cache(cache_key, info, expiration=86400)
        return info
    except Exception as e:
        print(f"❌ get_track_info error: {e}")
        return None


# -------------------------
# MusicBrainz -> MBID
# -------------------------
async def get_musicbrainz_id(track_name: str, artist_name: str) -> Optional[str]:
    cache_key = f"mbid:{track_name}:{artist_name}"
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached
    try:
        result = musicbrainzngs.search_recordings(recording=track_name, artist=artist_name, limit=1)
        recs = result.get("recording-list") or []
        if recs:
            mbid = recs[0].get("id")
            await cache_service.set_in_cache(cache_key, mbid, expiration=604800)
            return mbid
        return None
    except Exception as e:
        print(f"❌ MusicBrainz error: {e}")
        return None


# -------------------------
# AcousticBrainz -> Features
# -------------------------
async def get_audio_features_from_mbid(mbid: str) -> Optional[Dict]:
    cache_key = f"acousticbrainz:features:{mbid}"
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached
    try:
        url = f"{ACOUSTICBRAINZ_BASE_URL}/{mbid}/low-level"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        features = _normalize_acousticbrainz_features(data)
        _ensure_spec_rate(features)
        await cache_service.set_in_cache(cache_key, features, expiration=604800)
        return features
    except Exception as e:
        print(f"❌ AcousticBrainz error: {e}")
        return None


def _normalize_acousticbrainz_features(data: Dict) -> Dict:
    # Safe nested retrieval
    def _safe_get(d: Dict, *keys, default=0.5):
        try:
            curr = d
            for k in keys:
                if isinstance(curr, dict):
                    curr = curr.get(k, {})
                else:
                    return default
            if isinstance(curr, dict):
                # pick mean/value if present
                if "mean" in curr:
                    return float(curr["mean"])
                if "value" in curr:
                    return float(curr["value"])
                return default
            return float(curr)
        except Exception:
            return default

    try:
        features = {
            "valence": _safe_get(data, "highlevel", "mood_happy", "probability", default=0.5),
            "energy": _safe_get(data, "lowlevel", "dynamic_complexity", default=0.5),
            "danceability": _safe_get(data, "rhythm", "danceability", default=0.5),
            "acousticness": 1.0 - min(1.0, _safe_get(data, "lowlevel", "spectral_centroid", default=2000) / 4000),
            "instrumentalness": 1.0 - _safe_get(data, "highlevel", "voice_instrumental", "probability", default=0.5),
            "speechiness": min(1.0, _safe_get(data, "lowlevel", "spectral_rolloff", default=2000) / 8000),
            "tempo": _safe_get(data, "rhythm", "bpm", default=120.0),
            "loudness": _safe_get(data, "lowlevel", "loudness", default=-10.0),
            "liveness": min(1.0, _safe_get(data, "lowlevel", "average_loudness", default=0.2)),
            "key": int(_safe_get(data, "tonal", "key_key", default=0)) % 12,
            "mode": 1,
            "time_signature": 4,
            "duration_ms": int(_safe_get(data, "metadata", "audio_properties", "length", default=0) * 1000),
            "source": "acousticbrainz"
        }
        # Correct ranges
        for k in ["valence", "energy", "danceability", "acousticness", "instrumentalness", "speechiness", "liveness"]:
            features[k] = _clip(float(features[k]), 0.0, 1.0)
        features["tempo"] = _clip(float(features["tempo"]), 0.0, 300.0)
        features["loudness"] = _clip(float(features["loudness"]), -60.0, 0.0)
        features["key"] = int(_clip(int(features["key"]), 0, 11))
        return features
    except Exception as e:
        print(f"❌ Error normalizing AcousticBrainz: {e}")
        return get_default_features()


def get_default_features() -> Dict:
    base = {
        "danceability": 0.5, "energy": 0.5, "loudness": -10.0,
        "speechiness": 0.1, "acousticness": 0.5, "instrumentalness": 0.3,
        "liveness": 0.1, "valence": 0.5, "tempo": 120.0,
        "spec_rate": 0.5, "key": 0, "duration_ms": 0, "source": "default"
    }
    return base


# -------------------------
# High-level pipeline: get_audio_features (mbid -> acoustic -> gemini fallback)
# -------------------------
async def get_audio_features(
    track_name: str, 
    artist_name: str, 
    genre: Optional[str] = None, 
    use_gemini_fallback: bool = True,
    access_token: Optional[str] = None
) -> Dict:
    """
    Return a normalized audio feature dict for a track.
    Tries: Spotify API (if token provided) -> MusicBrainz MBID -> AcousticBrainz -> Gemini AI fallback -> defaults
    """
    cache_key = f"features:{track_name}:{artist_name}"
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached

    # Step 1: Try Spotify API (Preferred if token available)
    if access_token:
        try:
            print(f"🚀 Fetching audio features from Spotify: '{track_name}' by '{artist_name}'")
            # import spotify_service dynamically to avoid circular import issues
            from . import spotify_service
            spotify_tracks = await spotify_service.search_tracks(
                f"{track_name} {artist_name}",
                limit=1,
                access_token=access_token
            )
            if spotify_tracks:
                track_id = spotify_tracks[0]['id']
                features_list = await spotify_service.get_audio_features([track_id], access_token)
                if features_list and features_list[0]:
                    features = features_list[0]
                    # Ensure normalized features and source name
                    features['source'] = 'spotify'
                    _ensure_spec_rate(features)
                    # Cache for 1 week (highly reliable official data)
                    await cache_service.set_in_cache(cache_key, features, expiration=604800)
                    return features
        except Exception as e:
            print(f"⚠️ Spotify features fetch failed, falling back: {e}")

    # Step 2: Try MBID -> AcousticBrainz
    mbid = await get_musicbrainz_id(track_name, artist_name)
    if mbid:
        features = await get_audio_features_from_mbid(mbid)
        if features:
            # ensure spec_rate
            _ensure_spec_rate(features)
            await cache_service.set_in_cache(cache_key, features, expiration=604800)
            return features

    # Step 3: Gemini fallback
    if use_gemini_fallback:
        # try to find genre if missing
        if not genre:
            tags = await get_lastfm_tags(track_name, artist_name)
            genre = tags[0] if tags else None
        try:
            gemini_features = await gemini_service.estimate_audio_features_with_gemini(track_name, artist_name, genre=genre)
            if gemini_features:
                # Ensure normalized fields exist and spec_rate
                for k, default in get_default_features().items():
                    gemini_features.setdefault(k, default)
                _ensure_spec_rate(gemini_features)
                # cache shorter TTL for estimated features
                await cache_service.set_in_cache(cache_key, gemini_features, expiration=21600)
                return gemini_features
        except Exception as e:
            print(f"⚠️ Gemini fallback error: {e}")

    # Step 4: Final fallback
    fallback = get_default_features()
    _ensure_spec_rate(fallback)
    await cache_service.set_in_cache(cache_key, fallback, expiration=3600)
    return fallback


# -------------------------
# Playlist Aggregation (core)
# -------------------------
def aggregate_playlist_features(tracks: List[Dict]) -> Dict:
    """
    Compute accurate aggregated features for a playlist.

    - For each numeric feature, compute mean across tracks that have that feature.
    - For tempo/loudness conversion, preserve raw units; model_service.normalize_features handles mapping.
    - Returns dictionary containing aggregated features and diagnostics (counts, stddev, diversity)
    """
    # Track features per key
    feature_vals = {k: [] for k in FEATURE_KEYS}
    sources = {}

    for t in tracks:
        f = t.get("features") or t  # allow passing either enriched track or raw features
        if not f or not isinstance(f, dict):
            continue
        # record source
        sources[f.get("source", "unknown")] = sources.get(f.get("source", "unknown"), 0) + 1
        # gather
        for k in FEATURE_KEYS:
            if k in f and f[k] is not None:
                # Cast numbers safely
                try:
                    v = float(f[k])
                except Exception:
                    continue
                # collect
                feature_vals[k].append(v)

    aggregated = {}
    diagnostics = {}
    # compute mean & std and robust trimming for outliers (IQR trimming)
    for k, vals in feature_vals.items():
        if not vals:
            # if nothing available, use conservative default
            default = get_default_features().get(k, 0.0)
            aggregated[k] = default
            diagnostics[k] = {"count": 0, "std": 0.0}
            continue

        arr = np.array(vals, dtype=float)

        # Robust outlier trimming: remove extreme values outside 1.5*IQR
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            arr_trimmed = arr[(arr >= lower) & (arr <= upper)]
            if arr_trimmed.size >= max(1, int(0.5 * arr.size)):  # only trim if not too aggressive
                used = arr_trimmed
            else:
                used = arr  # fallback to full array if too much removed
        else:
            used = arr

        mean_val = float(np.mean(used))
        std_val = float(np.std(used))
        aggregated[k] = mean_val
        diagnostics[k] = {"count": int(arr.size), "used": int(used.size), "std": std_val}

    # Ensure spec_rate exists (map from tempo if needed)
    if "spec_rate" not in aggregated or aggregated.get("spec_rate") is None:
        aggregated["spec_rate"] = _normalize_spec_rate_from_tempo(aggregated.get("tempo", 120.0))

    # Add metadata
    aggregated["_metadata"] = {
        "track_count": len(tracks),
        "sources": sources,
        "diagnostics": diagnostics
    }
    return aggregated


# -------------------------
# Playlist Mood Prediction (uses model_service)
# -------------------------
async def predict_playlist_mood_from_aggregated(aggregated_features: Dict, top_k: int = 3) -> Dict:
    """
    Given aggregated_features (dict), call model_service.predict_mood_single_track
    to get up to top_k moods (model_service expects raw features mapping).
    """
    # model_service expects model_features keys as in MODEL_FEATURES; the model_service will normalize internals.
    # We call the model function and ensure structure.
    try:
        # model_service.predict_mood_single_track may be sync or async
        predictor = getattr(model_service, "predict_mood_single_track", None) or getattr(model_service, "predict_mood_from_features", None)
        if predictor is None:
            raise RuntimeError("model_service predictor not found")

        if asyncio.iscoroutinefunction(predictor):
            mood_data = await predictor(aggregated_features, top_k=top_k)
        else:
            mood_data = predictor(aggregated_features, top_k=top_k)

        # Ensure structure
        if not isinstance(mood_data, dict):
            raise RuntimeError("Invalid mood_data from model_service")
        return mood_data
    except Exception as e:
        print(f"⚠️ Playlist mood prediction error: {e}")
        # Fallback rule: simple valence/energy mapping
        val = aggregated_features.get("valence", 0.5)
        energy = aggregated_features.get("energy", 0.5)
        if energy > 0.7 and val > 0.5:
            primary = "Energetic"
        elif val > 0.6:
            primary = "Happy"
        elif val < 0.4:
            primary = "Sad"
        else:
            primary = "Calm"
        return {
            "primary_mood": primary,
            "all_moods": [primary],
            "mood_scores": {primary: 0.6},
            "confidence": 0.6,
            "source": "fallback_rule"
        }


# -------------------------
# Flow Optimizer
# -------------------------
def optimize_playlist_flow(tracks: List[Dict], progression: str = "gradual_rise") -> List[Dict]:
    """
    Reorder tracks to produce a smooth energy progression.
    - Primary key: energy (ascending for 'gradual_rise')
    - Secondary key: valence (ascending)
    - Then perform a smoothing pass to minimize adjacent energy jumps
    progression: 'gradual_rise' | 'gradual_fall' | 'steady' (steady = minimal reorder)
    """
    # Enrich features presence check
    enriched = []
    for t in tracks:
        f = t.get("features") or {}
        energy = float(f.get("energy", 0.5))
        valence = float(f.get("valence", 0.5))
        enriched.append((energy, valence, t))

    reverse = False
    if progression == "gradual_fall":
        reverse = True

    # Sort by energy then valence
    enriched_sorted = sorted(enriched, key=lambda x: (x[0], x[1]), reverse=reverse)
    ordered = [item[2] for item in enriched_sorted]

    # Smoothing pass: greedy adjacent swap to reduce large energy jumps
    def energy_of(track):
        f = track.get("features") or {}
        return float(f.get("energy", 0.5))

    # Perform a limited number of smoothing iterations
    max_iters = min(10, max(1, len(ordered)))
    for _ in range(max_iters):
        improved = False
        for i in range(len(ordered) - 1):
            e1 = energy_of(ordered[i])
            e2 = energy_of(ordered[i + 1])
            # If the jump is large, try swapping if it reduces sum of adjacent diffs
            cur_diff = abs(e1 - e2)
            # consider neighbor contexts
            prev = energy_of(ordered[i - 1]) if i - 1 >= 0 else None
            nxt = energy_of(ordered[i + 2]) if i + 2 < len(ordered) else None

            # compute total difference before and after swapping (local)
            before = cur_diff
            after = abs(energy_of(ordered[i]) - (nxt if nxt is not None else e1)) + abs((prev if prev is not None else e2) - energy_of(ordered[i + 1])) if (prev is not None or nxt is not None) else cur_diff

            # swap if we reduce max adjacent difference heuristically
            if cur_diff > 0.4:  # threshold for a "big" jump
                # simple heuristic: if swapping reduces neighbor variance, do it
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
                improved = True
        if not improved:
            break

    return ordered


# -------------------------
# Recommendations via MongoDB vector search (top-N)
# -------------------------
async def recommend_songs_by_playlist_vector(
    aggregated_features: Dict,
    limit: int = 20,
    collection_name: str = "databasesongs",
    vector_field: str = "audio_feature_vector",
    use_vector_search: bool = True,
    candidate_multiplier: int = 3
) -> List[Dict]:
    """
    Recommend songs by comparing aggregated_features vector against DB vectors.

    - Tries MongoDB $vectorSearch aggregation stage when available.
    - Fallback: fetch candidate set and compute cosine similarity in Python.
    - limit: number of results to return (10-50 normally).
    - aggregated_features: dict containing the same ordering of features as DB vectors.
    """
    # Build the query vector matching the DB vector format: required order should match how DB stored feature vectors.
    # We'll create 10-d normalized vector consistent with model_service.normalize_features expectations
    try:
        # Use model_service's normalization helper if available to get exact feature mapping
        normalize_fn = getattr(model_service, "normalize_features_for_vector", None)
    except Exception:
        normalize_fn = None

    # Construct a raw vector in a safe order for database: we will use the 10 model features in model_service.MODEL_FEATURES if available
    try:
        model_features = getattr(model_service, "MODEL_FEATURES", None)
        if model_features and isinstance(model_features, list) and len(model_features) >= 1:
            vector_order = model_features[:NUM_MODEL_FEATURES]
        else:
            # fallback to our best-guess order
            vector_order = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                            "instrumentalness", "liveness", "valence", "tempo", "spec_rate"]
    except Exception:
        vector_order = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                        "instrumentalness", "liveness", "valence", "tempo", "spec_rate"]

    # Compose numeric list for DB
    query_vector = []
    for k in vector_order:
        v = aggregated_features.get(k)
        if v is None:
            # fallback to defaults or estimate
            if k == "spec_rate":
                v = _normalize_spec_rate_from_tempo(aggregated_features.get("tempo", 120.0))
            else:
                v = get_default_features().get(k, 0.5)
        # If the DB stores normalized 0-1 inputs, convert loudness/tempo accordingly:
        if k == "loudness":
            # expected DB may store normalized loudness 0-1; convert -60..0 -> 0..1
            v = _clip((float(v) + 60.0) / 60.0, 0.0, 1.0)
        if k == "tempo":
            # normalize tempo 40-200 -> 0-1
            v = _clip((float(v) - 40.0) / 160.0, 0.0, 1.0)
        query_vector.append(float(v))

    # Try MongoDB vector search first
    try:
        db = get_mongo_client().get_default_database()
        coll = db[collection_name]
        if use_vector_search:
            # $vectorSearch is available on Atlas. We'll attempt an aggregation stage.
            pipeline = [
                {
                    "$search": {
                        "knnBeta": {
                            "vector": query_vector,
                            "path": vector_field,
                            "k": limit * candidate_multiplier
                        }
                    }
                },
                {"$limit": limit * candidate_multiplier},
                {"$project": {"score": {"$meta": "searchScore"}, "name": 1, "artists": 1, vector_field: 1, "metadata": 1}}
            ]
            # NOTE: some MongoDB deployments won't have $search stage; this will raise.
            cursor = coll.aggregate(pipeline)
            candidates = []
            async for doc in cursor:
                candidates.append(doc)
            # If we got candidates, compute cosine similarity optionally to re-rank
            if candidates:
                # compute cosine similarity and rank
                def _vec_from_doc(doc):
                    v = doc.get(vector_field) or doc.get("audio_feature_vector") or []
                    return np.array(v, dtype=float) if v else np.zeros(len(query_vector), dtype=float)

                qv = np.array(query_vector, dtype=float)
                scores = []
                for doc in candidates:
                    dv = _vec_from_doc(doc)
                    if dv.size != qv.size:
                        # try to resize or skip
                        continue
                    # cosine similarity
                    denom = (np.linalg.norm(qv) * np.linalg.norm(dv))
                    sim = float(np.dot(qv, dv) / denom) if denom > 0 else 0.0
                    scores.append((sim, doc))
                scores_sorted = sorted(scores, key=lambda x: x[0], reverse=True)[:limit]
                return [doc for _, doc in scores_sorted]

    except Exception as e:
        # Likely no Atlas Search / $vectorSearch available - fallback
        print(f"⚠️ MongoDB vector search not available or failed: {e}")

    # Fallback approach: coarse candidate fetch + in-memory cosine similarity.
    try:
        db = get_mongo_client().get_default_database()
        coll = db[collection_name]
        # Fetch candidates (limit to some reasonable number)
        cursor = coll.find({}, {vector_field: 1, "name": 1, "artists": 1, "metadata": 1}).limit(limit * candidate_multiplier)
        candidates = []
        async for doc in cursor:
            candidates.append(doc)

        if not candidates:
            return []

        qv = np.array(query_vector, dtype=float)
        scored = []
        for doc in candidates:
            dv = np.array(doc.get(vector_field) or doc.get("audio_feature_vector") or [], dtype=float)
            if dv.size != qv.size:
                continue
            denom = (np.linalg.norm(qv) * np.linalg.norm(dv))
            sim = float(np.dot(qv, dv) / denom) if denom > 0 else 0.0
            scored.append((sim, doc))
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:limit]
        return [doc for _, doc in top]

    except Exception as e:
        print(f"❌ Recommendation fallback failed: {e}")
        return []


# ============================================
# Last.fm Integration (Tags, Recommendations, Similar Artists)
# ============================================

async def get_similar_tracks_lastfm(track_name: str, artist_name: str, limit: int = 20) -> List[Dict]:
    """Get similar tracks for a track from Last.fm"""
    _init_lastfm()
    cache_key = f"lastfm:similar:{track_name}:{artist_name}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
        
    api_key = os.getenv("API_KEY_LASTFM") or os.getenv("LASTFM_API_KEY")
    if not api_key:
        return []
        
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getsimilar",
        "artist": artist_name,
        "track": track_name,
        "api_key": api_key,
        "format": "json",
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            similartracks = data.get('similartracks', {}).get('track', [])
            
            # Standardize format
            results = []
            for track in similartracks:
                # Last.fm returns artist as an object with #text or name
                artist_data = track.get('artist', {})
                artist_name_val = artist_data.get('name') if isinstance(artist_data, dict) else artist_data
                results.append({
                    'name': track.get('name'),
                    'artist': artist_name_val,
                    'match_score': float(track.get('match', 0))
                })
                
            await cache_service.set_in_cache(cache_key, results, expiration=604800) # 1 week
            return results
    except Exception as e:
        print(f"⚠️ Last.fm getsimilar failed: {e}")
        
    return []

async def get_similar_artists_lastfm(artist_name: str, limit: int = 10) -> List[str]:
    """Get similar artists for an artist from Last.fm"""
    _init_lastfm()
    cache_key = f"lastfm:similar_artists:{artist_name}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
        
    api_key = os.getenv("API_KEY_LASTFM") or os.getenv("LASTFM_API_KEY")
    if not api_key:
        return []
        
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getsimilar",
        "artist": artist_name,
        "api_key": api_key,
        "format": "json",
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            similarartists = data.get('similarartists', {}).get('artist', [])
            results = [artist.get('name') for artist in similarartists if artist.get('name')]
            await cache_service.set_in_cache(cache_key, results, expiration=604800)
            return results
    except Exception as e:
        print(f"⚠️ Last.fm artist.getsimilar failed: {e}")
        
    return []

async def get_recommendations(
    seed_track_name: str,
    seed_artist_name: str,
    target_mood: Optional[str] = None,
    limit: int = 20,
    access_token: Optional[str] = None
) -> List[Dict]:
    """
    Get track recommendations based on seed track and optional target mood.
    Uses Last.fm similar tracks as candidates and filters by target mood.
    """
    print(f"🎯 Getting recommendations based on seed: {seed_track_name} by {seed_artist_name} (mood: {target_mood})")
    
    # Get similar tracks from Last.fm
    similar_tracks = await get_similar_tracks_lastfm(seed_track_name, seed_artist_name, limit=limit * 3)
    if not similar_tracks:
        print("⚠️ No similar tracks from Last.fm, using seed as only option")
        similar_tracks = [{'name': seed_track_name, 'artist': seed_artist_name, 'match_score': 1.0}]
        
    # Enrich with audio features
    recommendations = []
    for track in similar_tracks[:limit]:
        # Get audio features (will use Gemini if needed!)
        features = await get_audio_features(
            track['name'],
            track['artist'],
            access_token=access_token
        )
        
        if features:
            enriched_track = {
                'name': track['name'],
                'artist': track['artist'],
                'features': features
            }
            
            # Predict mood
            # If target mood specified, filter
            if target_mood:
                # Simple heuristic mood filter for base moods
                mood_match = check_mood_match(features, target_mood)
                if not mood_match:
                    continue
            
            recommendations.append(enriched_track)
            
    print(f"✅ Generated {len(recommendations)} recommendations")
    return recommendations

def check_mood_match(features: Dict, target_mood: str) -> bool:
    """Check if features match target mood"""
    valence = features.get('valence', 0.5)
    energy = features.get('energy', 0.5)
    
    if target_mood == "Happy":
        return valence > 0.6 and energy > 0.5
    elif target_mood == "Sad":
        return valence < 0.4 and energy < 0.5
    elif target_mood == "Energetic":
        return energy > 0.7
    elif target_mood == "Calm":
        return energy < 0.4
    else:
        return True  # No filtering


# -------------------------
# Batch helpers
# -------------------------
async def batch_get_audio_features(tracks: List[Dict], concurrency: int = 5) -> List[Dict]:
    """
    Fetch audio features for multiple tracks concurrently; returns list of feature dicts.
    """
    sem = asyncio.Semaphore(concurrency)
    async def _fetch(t):
        async with sem:
            try:
                return await get_audio_features(t.get("name"), t.get("artist"))
            except Exception as e:
                print(f"⚠️ Error in batch_get_audio_features for {t}: {e}")
                return get_default_features()

    tasks = [asyncio.create_task(_fetch(t)) for t in tracks]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


# -------------------------
# Helper: enrich tracks with features & mood (useful for live queue)
# -------------------------
async def enrich_tracks_with_features_and_mood(tracks: List[Dict]) -> List[Dict]:
    """
    For each track in tracks (with keys: name, artist), fetch features and predict mood,
    then attach 'features' and 'mood' fields in-place and return the list.
    """
    enriched = []
    for t in tracks:
        name = t.get("name")
        artist = t.get("artist") or (t.get("artists")[0] if t.get("artists") else None)
        features = await get_audio_features(name, artist)
        t_out = dict(t)
        t_out["features"] = features
        # Predict mood (async/sync)
        mood = await predict_playlist_mood_from_aggregated(features, top_k=3)
        t_out["mood"] = mood
        enriched.append(t_out)
    return enriched


# -------------------------
# Tests & debug harness
# -------------------------
async def test_music_service():
    print("\n" + "="*60)
    print("🧪 Testing Music Service Flow")
    print("="*60)

    # Test single feature fetch
    tn = "Happy"
    ar = "Pharrell Williams"

    print("\n1) Get audio features (with possible Gemini fallback)")
    features = await get_audio_features(tn, ar)
    print("   Features sample:", {k: features.get(k) for k in ["valence", "energy", "tempo", "spec_rate", "source"]})

    print("\n2) Aggregate playlist (3 sample tracks)")
    sample_tracks = [
        {"name": "Happy", "artist": "Pharrell Williams"},
        {"name": "Someone Like You", "artist": "Adele"},
        {"name": "Smells Like Teen Spirit", "artist": "Nirvana"},
    ]
    # enrich features
    enriched = []
    for st in sample_tracks:
        f = await get_audio_features(st["name"], st["artist"])
        enriched.append({"name": st["name"], "artist": st["artist"], "features": f})
    agg = aggregate_playlist_features(enriched)
    print("   Aggregated (partial):", {k: agg.get(k) for k in ["danceability", "energy", "valence", "spec_rate"]})

    print("\n3) Predict playlist mood")
    mood = await predict_playlist_mood_from_aggregated(agg)
    print("   Mood:", mood.get("primary_mood"), "confidence:", mood.get("confidence"))

    print("\n4) Optimize playlist flow (calm->energetic)")
    ordered = optimize_playlist_flow(enriched, progression="gradual_rise")
    print("   Order (names):", [t["name"] for t in ordered])

    print("\n5) Recommend similar songs (top 10) - requires MongoDB & database songs")
    try:
        recs = await recommend_songs_by_playlist_vector(agg, limit=10)
        print("   Recommendation count:", len(recs))
        if recs:
            print("   Sample:", recs[0].get("name"), recs[0].get("artists"))
    except Exception as e:
        print(f"   Recommendation call failed: {e}")

    print("\n✅ Music service tests complete.")
    print("="*60)


# If run as a script
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_music_service())
