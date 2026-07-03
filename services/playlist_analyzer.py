"""
playlist_service.py

Phase 3 — Completed Playlist Service (Fusion-ready, analytics, optimizer, gap-filling)

Responsibilities:
- Robust playlist-level aggregation into 12 audio features (statistical + weighted)
- Playlist mood prediction (uses model_service.predict_mood_fused / predict_mood_single_track)
- Flow optimizer (smooth transitions, gap detection, gap-filling suggestions)
- Helpers for live-queue analytics (prepares aggregation, incremental update helper)
- Visualization-data preparation (line chart series + bar chart summary payloads)
- Compatibility with existing services: model_service, music_service, cache_service

Notes:
- This file purposely avoids changing signatures of other services.
- Expect model_service to expose:
    - MODEL_FEATURES (10 features)
    - MOOD_CLASSES (12 moods)
    - predict_mood_single_track(features, ...) -> dict
    - predict_mood_fused(features, lyric_hint, ...) -> dict or fused probs
- For recommendation/gap-filling, this file calls music_service.recommend_songs_by_playlist_vector()
  which should be present in your codebase (as in your provided music_service.py).
"""

from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict, Counter
import numpy as np
import asyncio
import math
from datetime import datetime

from . import model_service, music_service, cache_service

# Define the 12 features we want at playlist-level (final canonical set)
PLAYLIST_FEATURE_KEYS = [
    "danceability",
    "energy",
    "loudness",            # keep in raw dB at track-level, normalized at model input time
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",               # raw BPM
    "spec_rate",           # normalized spectral rate 0-1
    "key",                 # categorical (0-11)
    "duration_ms"
]

# For analytics and fusion we track also the 12 refined mood classes from model_service
MOOD_CLASSES = getattr(model_service, "MOOD_CLASSES", [
    "Happy","Sad","Energetic","Calm","Focused","Romantic",
    "Chill","Determined","Reflective","Confident","Anxious","Excited"
])

# Default fallbacks
DEFAULT_FEATURES = {
    "danceability": 0.5, "energy": 0.5, "loudness": -10.0,
    "speechiness": 0.1, "acousticness": 0.5, "instrumentalness": 0.3,
    "liveness": 0.1, "valence": 0.5, "tempo": 120.0, "spec_rate": 0.5,
    "key": 0, "duration_ms": 0
}

# Weights for aggregation importance (used for weighted averages)
FEATURE_IMPORTANCE = {
    "valence": 1.4,
    "energy": 1.4,
    "danceability": 1.0,
    "acousticness": 0.9,
    "instrumentalness": 0.8,
    "speechiness": 0.6,
    "tempo": 1.1,
    "loudness": 1.0,
    "liveness": 0.6,
    "spec_rate": 1.0,
    "key": 0.3,
    "duration_ms": 0.2
}

# tuning constants
IQR_MULTIPLIER = 1.5
SMOOTH_SWAP_ITERS = 12
ENERGY_JUMP_THRESHOLD = 0.35    # large jump marker for potential smoothing
GAP_FILL_CANDIDATES = 25        # number of candidate fills to fetch when gap detected


# -------------------------
# Utility helpers
# -------------------------

def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def _normalize_loudness_db_to_01(db: float) -> float:
    """Map -60..0 dB to 0..1"""
    return _clip((db + 60.0) / 60.0, 0.0, 1.0)

def _normalize_tempo_to_01(bpm: float) -> float:
    """Map 40..200 BPM to 0..1"""
    try:
        bpmf = float(bpm)
    except Exception:
        return 0.5
    return _clip((bpmf - 40.0) / 160.0, 0.0, 1.0)

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


# -------------------------
# Aggregation: robust & weighted
# -------------------------

def _iqr_trim(arr: np.ndarray) -> np.ndarray:
    """Perform IQR trimming; return trimmed array (or original if too aggressive)."""
    if arr.size == 0:
        return arr
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr <= 0:
        return arr
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    trimmed = arr[(arr >= lower) & (arr <= upper)]
    if trimmed.size < max(1, int(0.5 * arr.size)):
        return arr  # avoid over-trimming
    return trimmed


def aggregate_playlist_features_statistical(
    tracks: List[Dict],
    use_popularity_weighting: bool = True
) -> Dict[str, Any]:
    """
    High-quality playlist aggregation:
    - collects per-feature arrays, trims outliers with IQR,
    - calculates weighted mean using popularity and feature importance,
    - computes diagnostics (std, count, used_count) and diversity metrics,
    - returns aggregated features in original units (tempo in BPM, loudness in dB).
    """
    if not tracks:
        return {"_metadata": {"track_count": 0}, **DEFAULT_FEATURES}

    # Collect values
    collected = {k: [] for k in PLAYLIST_FEATURE_KEYS}
    popularity_weights = []

    # keep per-track energy for flow algorithms
    energies = []

    for i, t in enumerate(tracks):
        f = t.get("features") or {}
        # fallback: if top-level fields present on track object
        if not f:
            # try direct mapping if track dict contains feature keys
            f = {k: t.get(k) for k in PLAYLIST_FEATURE_KEYS if t.get(k) is not None}

        # popularity weight
        pop = _safe_float(t.get("popularity", t.get("pop", 50)))
        popularity_weights.append(_clip(pop/100.0, 0.0, 1.0))

        # gather features safely
        for k in PLAYLIST_FEATURE_KEYS:
            v = f.get(k, None)
            if v is None:
                # if tempo missing, try estimate from spec_rate if present
                if k == "tempo" and f.get("spec_rate") is not None:
                    # map back spec_rate -> tempo roughly
                    v = 40.0 + _clip(float(f["spec_rate"]), 0.0, 1.0) * 160.0
                else:
                    v = DEFAULT_FEATURES.get(k)
            # ensure numeric types
            if k in ("key", "duration_ms"):
                try:
                    collected[k].append(int(v))
                except Exception:
                    collected[k].append(int(DEFAULT_FEATURES.get(k, 0)))
            else:
                collected[k].append(float(v))
        energies.append(_safe_float(f.get("energy", 0.5)))

    # aggregated result
    aggregated = {}
    diagnostics = {}

    # Prepare popularity weights array
    pop_weights_arr = np.array(popularity_weights) if popularity_weights else None

    for k, vals in collected.items():
        arr = np.array(vals, dtype=float)
        if arr.size == 0:
            aggregated[k] = DEFAULT_FEATURES.get(k)
            diagnostics[k] = {"count": 0, "used": 0, "std": 0.0}
            continue

        # IQR trim for robustness
        used_arr = _iqr_trim(arr)

        # compute weights combining popularity & feature importance
        feat_weight = FEATURE_IMPORTANCE.get(k, 1.0)
        if use_popularity_weighting and pop_weights_arr is not None and pop_weights_arr.size == arr.size:
            weights = feat_weight * pop_weights_arr
            # If trimming removed elements, adapt weights
            if used_arr.size != arr.size:
                # map used indexes by filtering
                mask = np.isin(arr, used_arr)
                # if mask dims mismatch (repetitions), fallback to equal weights
                if mask.sum() == used_arr.size:
                    used_weights = weights[mask]
                else:
                    used_weights = np.ones_like(used_arr) * feat_weight
            else:
                used_weights = weights
            value = float(np.average(used_arr, weights=used_weights))
        else:
            # simple mean
            value = float(np.mean(used_arr))

        # keep original unit for loudness/tempo (we used raw values)
        aggregated[k] = value
        diagnostics[k] = {
            "count": int(arr.size),
            "used": int(used_arr.size),
            "std": float(np.std(used_arr))
        }

    # Derived / normalization helpers
    # Ensure spec_rate exists (derive from tempo if not accurate)
    if aggregated.get("spec_rate", None) is None or aggregated["spec_rate"] == 0:
        aggregated["spec_rate"] = _clip((aggregated.get("tempo", 120.0) - 40) / 160.0, 0.0, 1.0)

    # Add metadata
    aggregated["_metadata"] = {
        "track_count": len(tracks),
        "timestamp": datetime.utcnow().isoformat(),
        "diagnostics": diagnostics,
        "sources": Counter([ (t.get("features") or {}).get("source", "unknown") for t in tracks ])
    }

    return aggregated


# -------------------------
# Playlist mood prediction (single aggregated pass)
# -------------------------

async def predict_playlist_mood(
    tracks: List[Dict],
    use_lyrics_hint: bool = True,
    lyric_hint_provider: Optional[callable] = None,
    fusion_weights: Tuple[float, float] = (0.75, 0.25)
) -> Dict[str, Any]:
    """
    Given a list of tracks, produce a playlist-level mood prediction.
    Steps:
      1. Aggregate features using aggregate_playlist_features_statistical()
      2. Optionally collect lyric_hint (dictionary of mood probs) via lyric_hint_provider
      3. Use model_service.predict_mood_single_track() and optionally predict_mood_fused()
    Returns result with aggregated_features, mood prediction, and diagnostics.
    """
    aggregated = aggregate_playlist_features_statistical(tracks, use_popularity_weighting=True)

    # Build features dict that model_service expects (first 10 normalized fields retained)
    model_input = {k: aggregated.get(k, DEFAULT_FEATURES.get(k)) for k in [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "spec_rate"
    ]}

    # If loudness is stored as normalized 0..1 in some track sources, convert if needed:
    if 0.0 <= model_input["loudness"] <= 1.0:
        # try to detect if features came normalized (heuristic)
        # if tempo seems normalized (<=1) or loudness within -60..0 then convert.
        # We assume loudness in aggregated is raw dB unless clearly normalized.
        if aggregated.get("_metadata", {}).get("diagnostics", {}).get("loudness", {}).get("std", 0) < 0.001:
            # nothing to do, keep as-is
            pass
        else:
            # If loudness is in [0,1] but we expect dB, convert to dB for model_service normalization path
            model_input["loudness"] = model_input["loudness"] * -60.0  # convert 1->-60 (best-effort)
    # Call model predictions (sync function)
    try:
        base_prediction = model_service.predict_mood_single_track(model_input, top_k=3)
    except Exception as e:
        print(f"⚠️ model_service.predict_mood_single_track failed: {e}")
        base_prediction = {"primary_mood": "Unknown", "all_moods": [], "mood_scores": {}, "confidence": 0.0}

    lyric_hint = None
    if use_lyrics_hint and lyric_hint_provider:
        try:
            # lyric_hint_provider should accept tracks and return merged mood-prob dict matching MOOD_CLASSES
            lyric_hint = await lyric_hint_provider(tracks)
        except Exception as e:
            print(f"⚠️ lyric_hint_provider failed: {e}")
            lyric_hint = None

    if lyric_hint:
        # use model_service fusion
        fused_probs = model_service.predict_mood_fused(model_input, lyric_hint, weight_model=fusion_weights[0], weight_lyrics=fusion_weights[1])
        # build final prediction dict
        sorted_m = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
        top_k = [m for m, s in sorted_m[:3] if s > 0]
        if not top_k:
            top_k = [sorted_m[0][0]]
        result = {
            "primary_mood": top_k[0],
            "all_moods": top_k,
            "mood_scores": fused_probs,
            "confidence": float(sorted_m[0][1]),
            "source": "fusion_model"
        }
    else:
        result = base_prediction
        result["source"] = result.get("source", "model_service")

    # Attach aggregated features & metadata
    result["aggregated_features"] = aggregated
    result["track_count"] = len(tracks)
    return result


# -------------------------
# Flow optimizer (smooth order by energy & valence)
# -------------------------

def optimize_playlist_flow(
    tracks: List[Dict],
    progression: str = "gradual_rise",
    max_iters: int = SMOOTH_SWAP_ITERS
) -> List[Dict]:
    """
    Reorder playlist to create a smooth energy/valence progression.
    Heuristics:
      - Sort by energy primary (asc for gradual_rise, desc for gradual_fall)
      - Secondary sort by valence
      - Greedy smoothing swaps to reduce adjacent energy jumps
    Returns new ordered list (copy).
    """
    if not tracks:
        return []

    # Prepare list of tuples (energy, valence, original_index, track)
    enriched = []
    for idx, t in enumerate(tracks):
        f = t.get("features") or {}
        energy = _safe_float(f.get("energy", 0.5), 0.5)
        valence = _safe_float(f.get("valence", 0.5), 0.5)
        enriched.append((energy, valence, idx, t))

    reverse = False
    if progression == "gradual_fall":
        reverse = True

    enriched_sorted = sorted(enriched, key=lambda x: (x[0], x[1]), reverse=reverse)
    ordered = [tup[3] for tup in enriched_sorted]

    def energy_of(track):
        return _safe_float((track.get("features") or {}).get("energy", 0.5), 0.5)

    # smoothing pass
    iters = 0
    while iters < max_iters:
        improved = False
        for i in range(len(ordered) - 1):
            e1 = energy_of(ordered[i])
            e2 = energy_of(ordered[i+1])
            cur_gap = abs(e1 - e2)
            if cur_gap <= ENERGY_JUMP_THRESHOLD:
                continue
            # try swapping and measure neighbor variance
            before = 0.0
            after = 0.0
            # neighbors before swap
            prev_e = energy_of(ordered[i-1]) if i-1 >= 0 else None
            next_next_e = energy_of(ordered[i+2]) if i+2 < len(ordered) else None

            # compute local sum of adjacent diffs before and after
            def local_sum(arr_vals):
                s = 0.0
                for a, b in zip(arr_vals[:-1], arr_vals[1:]):
                    s += abs(a - b)
                return s

            seq_before = [x for x in [
                prev_e, e1, e2, next_next_e] if x is not None]
            seq_after = [x for x in [
                prev_e, e2, e1, next_next_e] if x is not None]

            before = local_sum(seq_before)
            after = local_sum(seq_after)

            # swap if reduces local adjacent diff
            if after < before:
                ordered[i], ordered[i+1] = ordered[i+1], ordered[i]
                improved = True
        if not improved:
            break
        iters += 1

    return ordered


# -------------------------
# Gap detection & gap-filling recommendations
# -------------------------

def detect_large_gaps(tracks: List[Dict], feature: str = "energy", threshold: float = 0.35) -> List[Dict]:
    """
    Scan playlist for large adjacent differences in a given feature (energy or valence).
    Returns list of gap descriptors:
      {
        "index": i,           # gap between i and i+1
        "left": track_i,
        "right": track_i+1,
        "left_value": ...,
        "right_value": ...,
        "gap": abs diff
      }
    """
    gaps = []
    for i in range(len(tracks)-1):
        left = tracks[i]
        right = tracks[i+1]
        lv = _safe_float((left.get("features") or {}).get(feature, 0.5))
        rv = _safe_float((right.get("features") or {}).get(feature, 0.5))
        gap = abs(lv - rv)
        if gap >= threshold:
            gaps.append({
                "index": i,
                "left": left,
                "right": right,
                "left_value": lv,
                "right_value": rv,
                "gap": gap
            })
    return gaps


async def suggest_gap_fillers(
    gap_descriptor: Dict,
    num_suggestions: int = 10,
    search_pool_multiplier: int = 2
) -> List[Dict]:
    """
    Suggest candidate tracks to fill a gap using music_service.recommend_songs_by_playlist_vector.
    Approach:
      - build a 'target' feature vector between left and right (midpoint)
      - query DB for nearest neighbors using music_service.recommend_songs_by_playlist_vector
    """
    left_f = gap_descriptor["left"].get("features") or {}
    right_f = gap_descriptor["right"].get("features") or {}

    # Build midpoint target features
    target = {}
    for k in PLAYLIST_FEATURE_KEYS:
        lv = left_f.get(k, DEFAULT_FEATURES.get(k))
        rv = right_f.get(k, DEFAULT_FEATURES.get(k))
        if k in ("key", "duration_ms"):
            try:
                target[k] = int(round((int(lv) + int(rv)) / 2))
            except Exception:
                target[k] = DEFAULT_FEATURES.get(k)
        else:
            target[k] = float(( _safe_float(lv) + _safe_float(rv) ) / 2.0)

    # Use music_service to recommend
    try:
        # aggregated model expects same shape as recommend_songs_by_playlist_vector
        candidates = await music_service.recommend_songs_by_playlist_vector(
            aggregated_features=target,
            limit=num_suggestions,
            candidate_multiplier=search_pool_multiplier
        )
        return candidates
    except Exception as e:
        print(f"⚠️ Gap filler recommendation failed: {e}")
        return []


# -------------------------
# Visualization helpers
# -------------------------

def prepare_playlist_visualization_data(tracks: List[Dict]) -> Dict[str, Any]:
    """
    Prepare aggregated chart-friendly payloads:
      - line_series: energy, valence, danceability, tempo across index
      - bar_summary: top mood distribution or averaged features
      - table_rows: per-track summary (name, artist, energy, valence, mood)
    """
    line_series = {
        "energy": [],
        "valence": [],
        "danceability": [],
        "tempo": []
    }
    table_rows = []
    mood_counts = Counter()

    for i, t in enumerate(tracks):
        f = t.get("features") or {}
        energy = _safe_float(f.get("energy", 0.5))
        valence = _safe_float(f.get("valence", 0.5))
        dance = _safe_float(f.get("danceability", 0.5))
        tempo = _safe_float(f.get("tempo", 120.0))
        # If tempo normalized to 0..1 convert to BPM heuristically
        if tempo <= 1.05:
            tempo_bpm = 40 + tempo * 160
        else:
            tempo_bpm = tempo

        line_series["energy"].append(energy)
        line_series["valence"].append(valence)
        line_series["danceability"].append(dance)
        line_series["tempo"].append(tempo_bpm)

        mood = t.get("mood") or (t.get("predicted_mood") or {}).get("primary_mood")
        if isinstance(mood, dict):
            mood = mood.get("primary_mood")
        if mood:
            mood_counts[mood] += 1

        table_rows.append({
            "index": i,
            "name": t.get("name"),
            "artist": (t.get("artists") or [{"name": t.get("artist")}])[0].get("name") if t.get("artists") else t.get("artist"),
            "energy": energy,
            "valence": valence,
            "danceability": dance,
            "tempo_bpm": tempo_bpm,
            "predicted_mood": mood
        })

    # build bar summary of averaged features & mood distribution
    avg_features = {}
    for k in ["energy", "valence", "danceability", "tempo"]:
        series = line_series.get(k, [])
        avg_features[k] = float(np.mean(series)) if series else 0.0

    # mood distribution percentages
    total_moods = sum(mood_counts.values()) or 1
    mood_distribution = {m: round((c/total_moods)*100, 2) for m, c in mood_counts.items()}
    top_moods = [m for m, _ in mood_counts.most_common(3)]

    return {
        "line_series": line_series,
        "avg_features": avg_features,
        "table_rows": table_rows,
        "mood_distribution": mood_distribution,
        "top_moods": top_moods
    }


# -------------------------
# Live queue incremental aggregation helpers
# -------------------------

async def incremental_live_queue_aggregate(
    redis_key: str,
    new_track: Dict,
    ttl_seconds: int = 300
) -> Dict[str, Any]:
    """
    When a live track starts playing, push its metadata to redis queue (cache_service).
    Then compute incremental aggregated features for live analytics.

    Expected cache_service interface:
      - append_to_list(key, value, expiration)
      - get_list(key) -> list
      - set_in_cache(key, value, expiration)
      - get_from_cache(key)

    Returns the new aggregated result for the live queue.
    """
    # get existing queue
    try:
        await cache_service.append_to_list(redis_key, new_track, expiration=ttl_seconds)
    except Exception:
        # fallback: try push/pop style
        try:
            lst = await cache_service.get_from_cache(redis_key) or []
            lst.append(new_track)
            await cache_service.set_in_cache(redis_key, lst, expiration=ttl_seconds)
        except Exception as e:
            print(f"⚠️ Live queue append fallback failed: {e}")

    # Read current queue and compute aggregate + mood
    current_queue = await cache_service.get_from_cache(redis_key) or []

    # If queue empty return early
    if not current_queue:
        return {"_metadata": {"track_count": 0}}

    # reuse aggregate function
    aggregated = aggregate_playlist_features_statistical(current_queue, use_popularity_weighting=False)

    # Predict mood quickly (no lyric hint in live)
    try:
        model_input = {k: aggregated.get(k, DEFAULT_FEATURES[k]) for k in [
            "danceability","energy","loudness","speechiness","acousticness",
            "instrumentalness","liveness","valence","tempo","spec_rate"
        ]}
        mood_prediction = model_service.predict_mood_single_track(model_input, top_k=3)
    except Exception as e:
        print(f"⚠️ Live mood prediction failed: {e}")
        mood_prediction = {"primary_mood": "Unknown", "all_moods": [], "mood_scores": {}, "confidence": 0.0}

    result = {
        "aggregated_features": aggregated,
        "mood": mood_prediction,
        "queue_length": len(current_queue),
        "last_updated": datetime.utcnow().isoformat()
    }

    # Optionally persist short-term analytics snapshot in cache
    try:
        await cache_service.set_in_cache(f"{redis_key}:last_snapshot", result, expiration=ttl_seconds*2)
    except Exception:
        pass

    return result


async def flush_live_queue_to_db(redis_key: str, user_id: Optional[str] = None, mongo_collection_name: str = "moodanalytics"):
    """
    When the live session ends (timeout or logout), flush the final aggregated snapshot to MongoDB.
    This function expects a backend API or DB helper; by default we will attempt to use
    cache_service.store_to_db or similar; if not available, we'll just return the payload.
    """
    snapshot = await cache_service.get_from_cache(f"{redis_key}:last_snapshot")
    if not snapshot:
        # compute from queue directly
        queue = await cache_service.get_from_cache(redis_key) or []
        snapshot = {
            "aggregated_features": aggregate_playlist_features_statistical(queue),
            "mood": {"primary_mood": "Unknown", "confidence": 0.0},
            "queue_length": len(queue),
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    snapshot["user_id"] = user_id
    snapshot["timestamp"] = datetime.utcnow().isoformat()

    # If cache_service exposes a DB write helper, use it
    try:
        writer = getattr(cache_service, "write_to_mongo", None)
        if callable(writer):
            await writer(mongo_collection_name, snapshot)
            return {"status": "stored", "collection": mongo_collection_name}
    except Exception as e:
        print(f"⚠️ flush_live_queue_to_db write failed: {e}")

    # Otherwise, return snapshot for the caller to persist
    return snapshot


# -------------------------
# Convenience test harness
# -------------------------

async def test_playlist_service_with_sample_tracks(sample_tracks: List[Dict]):
    """
    Quick test harness: aggregate, predict, optimize, detect gaps and prepare viz payload.
    """
    print("🧪 Testing playlist_service pipeline...")

    # 1) Aggregate
    agg = aggregate_playlist_features_statistical(sample_tracks)
    print("Aggregated (partial):", {k: agg[k] for k in ["danceability","energy","valence","tempo"]})

    # 2) Predict playlist mood
    pred = await predict_playlist_mood(sample_tracks, use_lyrics_hint=False)
    print("Playlist mood:", pred.get("primary_mood"), "confidence:", pred.get("confidence"))

    # 3) Optimize flow
    optimized = optimize_playlist_flow(sample_tracks, progression="gradual_rise")
    print("Optimized order:", [t.get("name") for t in optimized])

    # 4) Detect large gaps
    gaps = detect_large_gaps(optimized, feature="energy", threshold=0.35)
    print("Detected gaps:", len(gaps))
    if gaps:
        # suggest fillers for first gap (best-effort)
        suggestions = await suggest_gap_fillers(gaps[0], num_suggestions=5)
        print("Sample gap-fill suggestion count:", len(suggestions))

    # 5) Visual payload
    viz = prepare_playlist_visualization_data(sample_tracks)
    print("Visualization keys:", list(viz.keys()))

    return {
        "aggregated": agg,
        "prediction": pred,
        "optimized": optimized,
        "gaps": gaps,
        "viz": viz
    }


# Expose public API
class PlaylistAnalyzerWrapper:
    def aggregate_playlist_features(self, tracks: List[Dict]) -> Dict:
        res = aggregate_playlist_features_statistical(tracks)
        return res.get('mean', DEFAULT_FEATURES)

    def calculate_playlist_diversity(self, tracks: List[Dict]) -> Dict:
        if not tracks:
            return {"overall_diversity": 0.0}
        valences = []
        energies = []
        danceabilities = []
        for t in tracks:
            f = t.get("features") or {}
            valences.append(float(f.get("valence", 0.5)))
            energies.append(float(f.get("energy", 0.5)))
            danceabilities.append(float(f.get("danceability", 0.5)))
        std_val = float(np.std(valences)) if valences else 0.0
        std_en = float(np.std(energies)) if energies else 0.0
        std_dan = float(np.std(danceabilities)) if danceabilities else 0.0
        overall = float((std_val + std_en + std_dan) / 3.0)
        return {"overall_diversity": overall}

    def calculate_energy_progression(self, tracks: List[Dict]) -> Dict:
        if not tracks or len(tracks) < 2:
            return {"progression_type": "flat", "trend": 0.0}
        energies = []
        for t in tracks:
            f = t.get("features") or {}
            energies.append(float(f.get("energy", 0.5)))
        x = np.arange(len(energies))
        slope, _ = np.polyfit(x, energies, 1)
        if slope > 0.05:
            prog = "rising"
        elif slope < -0.05:
            prog = "falling"
        else:
            prog = "flat"
        return {"progression_type": prog, "trend": float(slope)}

    def predict_playlist_mood(self, *args, **kwargs):
        return predict_playlist_mood(*args, **kwargs)
    def optimize_playlist_flow(self, *args, **kwargs):
        return optimize_playlist_flow(*args, **kwargs)
    def detect_large_gaps(self, *args, **kwargs):
        return detect_large_gaps(*args, **kwargs)
    def suggest_gap_fillers(self, *args, **kwargs):
        return suggest_gap_fillers(*args, **kwargs)
    def prepare_playlist_visualization_data(self, *args, **kwargs):
        return prepare_playlist_visualization_data(*args, **kwargs)

playlist_analyzer = PlaylistAnalyzerWrapper()

__all__ = [
    "playlist_analyzer",
    "aggregate_playlist_features_statistical",
    "predict_playlist_mood",
    "optimize_playlist_flow",
    "detect_large_gaps",
    "suggest_gap_fillers",
    "prepare_playlist_visualization_data",
    "incremental_live_queue_aggregate",
    "flush_live_queue_to_db",
    "test_playlist_service_with_sample_tracks"
]
