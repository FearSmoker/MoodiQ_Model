"""
Enhanced Model Service - 12 Moods with Multi-Tag Classification
- Maintains 4 base moods from trained model
- Maps to 12 extended moods algorithmically
- Returns 2-3 tags per song based on feature similarity (70-80% threshold)
"""

import numpy as np
import onnxruntime as ort
import os
import json
from typing import Dict, List, Optional, Tuple
from . import cache_service
from . import music_service
from . import spotify_service as sp_service
from .playlist_analyzer import playlist_analyzer

# Model Configuration
MOOD_MODEL_PATH = os.path.join("models", "mood_model.onnx")

def load_mood_classes():
    """Load mood classes from metadata"""
    metadata_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('mood_classes', ["Calm", "Energetic", "Happy", "Sad"])
    return ["Calm", "Energetic", "Happy", "Sad"]

# Base moods from trained model
BASE_MOOD_CLASSES = load_mood_classes()
print(f"🎭 Base mood classes: {BASE_MOOD_CLASSES}")

# Extended 12 moods with feature profiles
EXTENDED_MOODS = {
    "Relaxed": {
        "base_moods": ["Calm"],
        "profile": {
            "valence": 0.5, "energy": 0.25, "danceability": 0.3,
            "acousticness": 0.7, "tempo": 80, "loudness": -15
        },
        "weights": {"valence": 0.8, "energy": 1.0, "acousticness": 0.9}
    },
    "Focused": {
        "base_moods": ["Calm"],
        "profile": {
            "valence": 0.45, "energy": 0.4, "danceability": 0.35,
            "acousticness": 0.5, "speechiness": 0.05, "instrumentalness": 0.6
        },
        "weights": {"energy": 1.0, "speechiness": 0.9, "instrumentalness": 0.8}
    },
    "Romantic": {
        "base_moods": ["Calm", "Happy"],
        "profile": {
            "valence": 0.6, "energy": 0.35, "danceability": 0.4,
            "acousticness": 0.6, "tempo": 90
        },
        "weights": {"valence": 0.9, "energy": 0.8, "acousticness": 0.7}
    },
    "Excited": {
        "base_moods": ["Happy", "Energetic"],
        "profile": {
            "valence": 0.8, "energy": 0.85, "danceability": 0.8,
            "tempo": 140, "loudness": -4
        },
        "weights": {"valence": 0.9, "energy": 1.0, "danceability": 0.9}
    },
    "Angry": {
        "base_moods": ["Energetic", "Sad"],
        "profile": {
            "valence": 0.25, "energy": 0.85, "danceability": 0.5,
            "loudness": -3, "tempo": 130
        },
        "weights": {"valence": 1.0, "energy": 1.0, "loudness": 0.9}
    },
    "Chill": {
        "base_moods": ["Calm", "Happy"],
        "profile": {
            "valence": 0.6, "energy": 0.3, "danceability": 0.45,
            "acousticness": 0.55, "tempo": 95
        },
        "weights": {"energy": 1.0, "valence": 0.8, "danceability": 0.7}
    },
    "Melancholic": {
        "base_moods": ["Sad"],
        "profile": {
            "valence": 0.2, "energy": 0.25, "danceability": 0.3,
            "acousticness": 0.7, "tempo": 70
        },
        "weights": {"valence": 1.0, "energy": 0.9, "acousticness": 0.8}
    },
    "Dreamy": {
        "base_moods": ["Calm", "Sad"],
        "profile": {
            "valence": 0.4, "energy": 0.3, "danceability": 0.35,
            "acousticness": 0.65, "tempo": 85, "instrumentalness": 0.5
        },
        "weights": {"energy": 0.9, "acousticness": 1.0, "instrumentalness": 0.8}
    },
    "Motivated": {
        "base_moods": ["Energetic"],
        "profile": {
            "valence": 0.65, "energy": 0.75, "danceability": 0.65,
            "tempo": 125, "loudness": -6
        },
        "weights": {"energy": 1.0, "valence": 0.8, "tempo": 0.7}
    },
    "Joyful": {
        "base_moods": ["Happy"],
        "profile": {
            "valence": 0.85, "energy": 0.7, "danceability": 0.75,
            "tempo": 120, "loudness": -5
        },
        "weights": {"valence": 1.0, "energy": 0.8, "danceability": 0.9}
    },
    "Ambient": {
        "base_moods": ["Calm"],
        "profile": {
            "valence": 0.5, "energy": 0.2, "danceability": 0.25,
            "instrumentalness": 0.9, "acousticness": 0.6, "tempo": 70
        },
        "weights": {"instrumentalness": 1.0, "energy": 0.9, "speechiness": 0.8}
    },
    "Party": {
        "base_moods": ["Energetic", "Happy"],
        "profile": {
            "valence": 0.8, "energy": 0.9, "danceability": 0.9,
            "tempo": 128, "loudness": -4
        },
        "weights": {"danceability": 1.0, "energy": 1.0, "valence": 0.9}
    }
}

# All mood labels (base + extended)
ALL_MOOD_LABELS = list(EXTENDED_MOODS.keys())
print(f"🎵 Extended moods: {ALL_MOOD_LABELS}")

MODEL_FEATURE_ORDER = [
    'valence', 'energy', 'danceability', 'acousticness', 
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

# Genre weights for lyrics vs audio
GENRE_WEIGHTS = {
    'pop': {'audio': 0.6, 'lyrics': 0.4},
    'rock': {'audio': 0.7, 'lyrics': 0.3},
    'hip-hop': {'audio': 0.4, 'lyrics': 0.6},
    'rap': {'audio': 0.35, 'lyrics': 0.65},
    'electronic': {'audio': 0.85, 'lyrics': 0.15},
    'edm': {'audio': 0.85, 'lyrics': 0.15},
    'classical': {'audio': 0.95, 'lyrics': 0.05},
    'indie': {'audio': 0.5, 'lyrics': 0.5},
    'r&b': {'audio': 0.5, 'lyrics': 0.5},
    'rnb': {'audio': 0.5, 'lyrics': 0.5},
    'country': {'audio': 0.5, 'lyrics': 0.5},
    'jazz': {'audio': 0.85, 'lyrics': 0.15},
    'default': {'audio': 0.65, 'lyrics': 0.35}
}

# Model globals
mood_model = None
session_options = None
scaler_mean = None
scaler_scale = None


def load_model():
    """Load ONNX model"""
    global mood_model, session_options, scaler_mean, scaler_scale
    
    try:
        if not os.path.exists(MOOD_MODEL_PATH):
            print(f"⚠️ Model not found. Using rule-based fallback.")
            return
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        mood_model = ort.InferenceSession(
            MOOD_MODEL_PATH, 
            session_options,
            providers=['CPUExecutionProvider']
        )
        
        print(f"✅ ONNX model loaded")
        
        metadata_path = os.path.join("models", "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                scaler_mean = np.array(metadata['scaler_mean'], dtype=np.float32)
                scaler_scale = np.array(metadata['scaler_scale'], dtype=np.float32)
                print(f"✅ Metadata loaded")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}. Using rule-based fallback.")
        mood_model = None


def normalize_features(features: Dict) -> np.ndarray:
    """Normalize audio features using saved scaler"""
    feature_values = []
    for feature_name in MODEL_FEATURE_ORDER:
        value = features.get(feature_name, 0.5)
        feature_values.append(float(value))
    
    feature_array = np.array(feature_values, dtype=np.float32)
    
    if scaler_mean is not None and scaler_scale is not None:
        feature_array = (feature_array - scaler_mean) / scaler_scale
        feature_array = feature_array.astype(np.float32)
    
    return feature_array


def calculate_mood_similarity(features: Dict, mood_name: str) -> float:
    """
    Calculate similarity between track features and mood profile.
    Returns similarity score (0-1).
    """
    mood_profile = EXTENDED_MOODS[mood_name]["profile"]
    weights = EXTENDED_MOODS[mood_name]["weights"]
    
    total_similarity = 0.0
    total_weight = 0.0
    
    # Compare each weighted feature
    for feature, target_value in mood_profile.items():
        if feature not in features:
            continue
        
        actual_value = features[feature]
        weight = weights.get(feature, 0.5)
        
        # Normalize feature values for comparison
        if feature == 'tempo':
            # Tempo: normalize to 0-1 range (40-200 BPM)
            actual_normalized = (actual_value - 40) / 160
            target_normalized = (target_value - 40) / 160
        elif feature == 'loudness':
            # Loudness: normalize to 0-1 range (-60 to 0 dB)
            actual_normalized = (actual_value + 60) / 60
            target_normalized = (target_value + 60) / 60
        else:
            # Already normalized (0-1)
            actual_normalized = actual_value
            target_normalized = target_value
        
        # Calculate distance (1 - distance = similarity)
        distance = abs(actual_normalized - target_normalized)
        similarity = 1.0 - distance
        
        total_similarity += similarity * weight
        total_weight += weight
    
    # Weighted average similarity
    if total_weight > 0:
        return total_similarity / total_weight
    
    return 0.0


def get_multi_mood_tags(
    features: Dict,
    min_similarity: float = 0.70,
    max_tags: int = 3
) -> List[Tuple[str, float]]:
    """
    Get multiple mood tags for a track based on feature similarity.
    
    Args:
        features: Audio features dictionary
        min_similarity: Minimum similarity threshold (0.70 = 70%)
        max_tags: Maximum number of tags to return (default: 3)
    
    Returns:
        List of (mood, similarity_score) tuples
    """
    mood_scores = []
    
    # Calculate similarity for each extended mood
    for mood_name in ALL_MOOD_LABELS:
        similarity = calculate_mood_similarity(features, mood_name)
        
        if similarity >= min_similarity:
            mood_scores.append((mood_name, similarity))
    
    # Sort by similarity (highest first)
    mood_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N moods
    return mood_scores[:max_tags]


async def predict_mood_from_spotify_track(
    track_id: str,
    access_token: str,
    lyrics_sentiment: Dict,
    user_id: Optional[str] = None
) -> Dict:
    """
    HYBRID APPROACH: Predict mood(s) for Spotify track with multi-tag support
    """
    # Check user override first
    if user_id and track_id:
        override_key = f"user_model:{user_id}:track:{track_id}"
        cached_override = await cache_service.get_from_cache(override_key)
        if cached_override:
            return {
                "track_id": track_id,
                "primary_mood": cached_override,
                "all_moods": [cached_override],
                "mood_scores": {cached_override: 1.0},
                "confidence": 1.0,
                "source": "user_feedback"
            }
    
    # Get track metadata from Spotify
    track_info = await sp_service.get_track_info(track_id, access_token)
    
    if not track_info:
        raise ValueError(f"Track {track_id} not found on Spotify")
    
    track_name = track_info['name']
    artist_name = sp_service.get_primary_artist_name(track_info)
    
    print(f"🎵 Analyzing: {track_name} by {artist_name}")
    
    # Get audio features
    audio_features = None
    
    if os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET"):
        try:
            sp_client = sp_service.get_spotify_client(access_token)
            spotify_features = sp_client.audio_features([track_id])
            
            if spotify_features and spotify_features[0]:
                audio_features = spotify_features[0]
                print(f"✅ Using Spotify audio features")
        except Exception as e:
            print(f"⚠️ Spotify audio features failed: {e}")
    
    if not audio_features:
        print(f"⚠️ Falling back to multi-API feature extraction")
        audio_features = await music_service.get_audio_features(track_name, artist_name)
    
    if not audio_features:
        print("⚠️ Using default audio features")
        audio_features = music_service.get_default_features()
    
    # Get genre tags
    tags = await music_service.get_lastfm_tags(track_name, artist_name)
    genre = tags[0].lower() if tags else None
    
    # Predict mood(s)
    mood_data = await predict_mood_from_features(
        audio_features,
        lyrics_sentiment,
        user_id=user_id,
        track_id=track_id,
        genre=genre
    )
    
    # Add Spotify metadata
    mood_data['track_info'] = {
        'id': track_info['id'],
        'name': track_info['name'],
        'artists': [a['name'] for a in track_info['artists']],
        'album': track_info['album']['name'],
        'popularity': track_info['popularity'],
        'duration_ms': track_info['duration_ms'],
        'external_url': track_info.get('external_url')
    }
    
    return mood_data


async def predict_mood_from_features(
    audio_features: Dict, 
    lyrics_sentiment: Dict, 
    user_id: Optional[str] = None,
    track_id: Optional[str] = None,
    genre: Optional[str] = None
) -> Dict:
    """
    Core mood prediction with multi-tag support (12 moods, 2-3 tags)
    """
    
    valence = float(audio_features.get('valence', 0.5))
    energy = float(audio_features.get('energy', 0.5))
    danceability = float(audio_features.get('danceability', 0.5))
    
    print(f"📊 Audio features - Valence: {valence:.2f}, Energy: {energy:.2f}, Dance: {danceability:.2f}")

    # 1. Get base mood from ONNX model
    base_mood = "Calm"
    base_confidence = 0.0
    base_probabilities = None
    
    if mood_model:
        try:
            input_data = normalize_features(audio_features)
            input_data = input_data.reshape(1, -1).astype(np.float32)
            
            input_name = mood_model.get_inputs()[0].name
            nn_output = mood_model.run(None, {input_name: input_data})[0]
            base_probabilities = nn_output[0]
            
            # Apply personalized adjustments
            if user_id:
                base_probabilities = await apply_personalized_adjustments(
                    audio_features, base_probabilities, user_id
                )
            
            predicted_index = np.argmax(base_probabilities)
            base_confidence = float(base_probabilities[predicted_index])
            base_mood = BASE_MOOD_CLASSES[predicted_index]
            
            print(f"🤖 Base model: {base_mood} ({base_confidence:.2%})")

        except Exception as e:
            print(f"⚠️ Model prediction failed: {e}")
            base_mood, base_confidence = _rule_based_mood(valence, energy, danceability)
    else:
        base_mood, base_confidence = _rule_based_mood(valence, energy, danceability)
    
    # 2. Get multi-mood tags (12 extended moods)
    multi_moods = get_multi_mood_tags(
        audio_features,
        min_similarity=0.70,  # 70% threshold
        max_tags=3
    )
    
    # 3. Adjust with lyrics sentiment
    lyric_polarity = float(lyrics_sentiment.get('polarity', 0.0))
    lyric_subjectivity = float(lyrics_sentiment.get('subjectivity', 0.0))
    
    print(f"📝 Lyrics - Polarity: {lyric_polarity:.2f}, Subjectivity: {lyric_subjectivity:.2f}")
    
    # Genre-adaptive fusion
    weights = GENRE_WEIGHTS.get(genre if genre else 'default', GENRE_WEIGHTS['default'])
    lyric_strength = abs(lyric_polarity) * lyric_subjectivity
    
    if lyric_strength > 0.5:
        lyric_weight = min(weights['lyrics'] * (1 + lyric_strength), 0.8)
    else:
        lyric_weight = weights['lyrics'] * lyric_strength
    
    audio_weight = 1.0 - lyric_weight
    
    # Determine lyrics mood influence
    if lyric_polarity > 0.3:
        lyrics_mood = "Positive"
    elif lyric_polarity < -0.3:
        lyrics_mood = "Negative"
    else:
        lyrics_mood = "Neutral"
    
    # Build response
    all_mood_names = [mood for mood, score in multi_moods]
    mood_scores = {mood: float(score) for mood, score in multi_moods}
    
    # If no moods matched 70% threshold, use base mood
    if not all_mood_names:
        all_mood_names = [base_mood]
        mood_scores = {base_mood: base_confidence}
    
    primary_mood = all_mood_names[0]
    
    print(f"🎯 Final moods: {', '.join(all_mood_names)}")
    
    return {
        "primary_mood": primary_mood,
        "all_moods": all_mood_names,
        "mood_scores": mood_scores,
        "base_mood": base_mood,
        "lyrics_mood": lyrics_mood,
        "confidence": mood_scores.get(primary_mood, base_confidence),
        "source": "ml_model_multi_tag" if mood_model else "rule_based_multi_tag",
        "scores": {
            "valence": valence,
            "energy": energy,
            "danceability": danceability,
            "acousticness": float(audio_features.get('acousticness', 0.5)),
            "lyrics_polarity": lyric_polarity,
            "lyrics_subjectivity": lyric_subjectivity,
            "audio_weight": audio_weight,
            "lyric_weight": lyric_weight
        },
        "genre": genre,
        "base_model_probabilities": dict(zip(BASE_MOOD_CLASSES, base_probabilities)) if base_probabilities is not None else None,
        "num_tags": len(all_mood_names)
    }

async def predict_playlist_mood(
    tracks: List[Dict],
    user_id: Optional[str] = None
) -> Dict:
    """
    Predict mood for entire playlist using advanced aggregation
    
    Args:
        tracks: List of track dictionaries with 'features' key
        user_id: Optional user ID for personalization
        
    Returns:
        Mood prediction for entire playlist
    """
    if not tracks:
        return {
            'primary_mood': 'Unknown',
            'all_moods': [],
            'mood_scores': {},
            'confidence': 0.0,
            'track_count': 0
        }
    
    print(f"🎵 Analyzing playlist: {len(tracks)} tracks")
    
    # Use advanced aggregation
    aggregated_features = playlist_analyzer.aggregate_playlist_features(
        tracks,
        use_popularity_weighting=True
    )
    
    # Get diversity metrics
    diversity = playlist_analyzer.calculate_playlist_diversity(tracks)
    
    # Get energy progression
    progression = playlist_analyzer.calculate_energy_progression(tracks)
    
    print(f"   Aggregated features calculated")
    print(f"   Diversity score: {diversity['overall_diversity']:.3f}")
    print(f"   Energy progression: {progression['progression_type']}")
    
    # Predict mood for aggregated features
    mood_data = await predict_mood_from_features(
        aggregated_features,
        lyrics_sentiment={'polarity': 0.0, 'subjectivity': 0.0},
        user_id=user_id
    )
    
    # Add playlist-specific metadata
    mood_data['playlist_analysis'] = {
        'track_count': len(tracks),
        'diversity_metrics': diversity,
        'energy_progression': progression,
        'aggregation_metadata': aggregated_features.get('_metadata', {})
    }
    
    print(f"✅ Playlist mood: {mood_data['primary_mood']}")
    print(f"   All moods: {', '.join(mood_data['all_moods'])}")
    
    return mood_data


# Add this helper function for flow optimization

def calculate_playlist_smoothness(tracks: List[Dict]) -> float:
    """
    Calculate how smooth the playlist flow is (0-1)
    
    Higher score = smoother transitions
    
    Args:
        tracks: List of tracks in order
        
    Returns:
        Smoothness score
    """
    if len(tracks) < 2:
        return 1.0
    
    total_distance = 0.0
    transitions = 0
    
    for i in range(len(tracks) - 1):
        current_features = tracks[i].get('features', {})
        next_features = tracks[i + 1].get('features', {})
        
        if not current_features or not next_features:
            continue
        
        # Calculate Euclidean distance in feature space
        distance = 0.0
        feature_count = 0
        
        for feature in ['valence', 'energy', 'danceability']:
            val1 = current_features.get(feature, 0.5)
            val2 = next_features.get(feature, 0.5)
            distance += (val1 - val2) ** 2
            feature_count += 1
        
        if feature_count > 0:
            distance = np.sqrt(distance / feature_count)
            total_distance += distance
            transitions += 1
    
    if transitions == 0:
        return 1.0
    
    # Average distance (lower is smoother)
    avg_distance = total_distance / transitions
    
    # Convert to smoothness score (0-1, higher is better)
    # Max expected distance is ~1.732 (sqrt of 3) if all features differ by 1.0
    smoothness = max(0, 1 - (avg_distance / 1.732))
    
    return float(smoothness)


# Add this to enhance the optimize_flow_dp function

def optimize_flow_with_gradient(
    tracks: List[Dict],
    target_start_energy: float = 0.3,
    target_end_energy: float = 0.8,
    smoothness_weight: float = 0.7
) -> Dict:
    """
    Optimize playlist flow with energy gradient (calm to energetic)
    
    Args:
        tracks: List of track dictionaries
        target_start_energy: Desired energy at start (0-1)
        target_end_energy: Desired energy at end (0-1)
        smoothness_weight: How much to prioritize smoothness (0-1)
        
    Returns:
        Optimized order with flow score
    """
    n = len(tracks)
    
    if n == 0:
        return {"optimizedOrder": [], "flowScore": 0, "transitions": []}
    
    if n == 1:
        return {"optimizedOrder": [0], "flowScore": 1.0, "transitions": []}
    
    print(f"🔄 Optimizing flow: {n} tracks")
    print(f"   Target: {target_start_energy:.2f} → {target_end_energy:.2f} energy")
    
    # Extract energy values
    track_energies = []
    for track in tracks:
        features = track.get('features', {})
        energy = features.get('energy', 0.5)
        track_energies.append((tracks.index(track), energy))
    
    # Sort by energy
    sorted_by_energy = sorted(track_energies, key=lambda x: x[1])
    
    # Calculate ideal energy gradient
    energy_gradient = np.linspace(target_start_energy, target_end_energy, n)
    
    # Match tracks to gradient positions
    assigned = []
    used_indices = set()
    
    for target_energy in energy_gradient:
        # Find closest available track
        best_idx = None
        best_distance = float('inf')
        
        for idx, energy in sorted_by_energy:
            if idx in used_indices:
                continue
            
            distance = abs(energy - target_energy)
            
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        
        if best_idx is not None:
            assigned.append(best_idx)
            used_indices.add(best_idx)
    
    # Calculate smoothness
    optimized_tracks = [tracks[i] for i in assigned]
    smoothness = calculate_playlist_smoothness(optimized_tracks)
    
    # Calculate energy gradient adherence
    actual_energies = [tracks[i].get('features', {}).get('energy', 0.5) for i in assigned]
    gradient_error = np.mean(np.abs(np.array(actual_energies) - energy_gradient))
    gradient_score = max(0, 1 - gradient_error)
    
    # Combined flow score
    flow_score = (smoothness_weight * smoothness) + ((1 - smoothness_weight) * gradient_score)
    
    # Generate transitions
    transitions = []
    for i in range(len(assigned) - 1):
        curr_idx = assigned[i]
        next_idx = assigned[i + 1]
        
        curr_features = tracks[curr_idx].get('features', {})
        next_features = tracks[next_idx].get('features', {})
        
        curr_energy = curr_features.get('energy', 0.5)
        next_energy = next_features.get('energy', 0.5)
        
        energy_diff = next_energy - curr_energy
        
        transitions.append({
            "from_index": int(curr_idx),
            "to_index": int(next_idx),
            "energy_from": float(curr_energy),
            "energy_to": float(next_energy),
            "energy_change": float(energy_diff),
            "smoothness": float(smoothness)
        })
    
    print(f"✅ Flow optimized: score = {flow_score:.3f}")
    print(f"   Smoothness: {smoothness:.3f}")
    print(f"   Gradient adherence: {gradient_score:.3f}")
    
    return {
        "optimizedOrder": assigned,
        "flowScore": float(flow_score),
        "transitions": transitions,
        "smoothness": float(smoothness),
        "gradient_score": float(gradient_score),
        "energy_start": float(actual_energies[0]) if actual_energies else 0.0,
        "energy_end": float(actual_energies[-1]) if actual_energies else 0.0
    }

async def apply_personalized_adjustments(
    features: Dict,
    mood_probabilities: np.ndarray,
    user_id: Optional[str]
) -> np.ndarray:
    """Apply user-specific adjustments"""
    if not user_id:
        return mood_probabilities
    
    user_model_key = f"user_model:{user_id}:trained"
    user_model = await cache_service.get_from_cache(user_model_key)
    
    if not user_model:
        return mood_probabilities
    
    mood_weights = user_model.get('mood_weights', {})
    adjusted_probs = mood_probabilities.copy()
    
    for i, mood in enumerate(BASE_MOOD_CLASSES):
        if mood in mood_weights:
            boost = mood_weights[mood] * 0.3
            adjusted_probs[i] = adjusted_probs[i] * (1 + boost)
    
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    return adjusted_probs


def _rule_based_mood(valence: float, energy: float, danceability: float) -> tuple:
    """Rule-based fallback"""
    if valence > 0.6 and energy > 0.6:
        return "Happy", 0.75
    elif energy > 0.7:
        return "Energetic", 0.70
    elif energy < 0.4 and valence > 0.4:
        return "Calm", 0.70
    elif energy < 0.4 and valence < 0.4:
        return "Sad", 0.75
    elif valence < 0.3:
        return "Sad", 0.65
    else:
        return "Calm", 0.50


def map_external_mood_to_extended(external_mood: str) -> str:
    """Map external mood to one of the 12 extended moods"""
    external_mood_lower = external_mood.lower().strip()
    
    # Direct match
    for mood in ALL_MOOD_LABELS:
        if external_mood_lower == mood.lower():
            return mood
    
    # Synonym mapping
    mood_mapping = {
        'happy': 'Joyful', 'joyful': 'Joyful', 'cheerful': 'Joyful',
        'sad': 'Melancholic', 'melancholy': 'Melancholic', 'depressed': 'Melancholic',
        'calm': 'Relaxed', 'peaceful': 'Relaxed', 'chill': 'Chill',
        'energetic': 'Motivated', 'hyped': 'Excited', 'intense': 'Excited',
        'workout': 'Motivated', 'party': 'Party', 'study': 'Focused',
        'romantic': 'Romantic', 'angry': 'Angry', 'ambient': 'Ambient',
        'dreamy': 'Dreamy', 'motivated': 'Motivated'
    }
    
    if external_mood_lower in mood_mapping:
        return mood_mapping[external_mood_lower]
    
    # Default to Relaxed
    return 'Relaxed'


def optimize_flow_dp(tracks: List[Dict], start_mood: Dict, end_mood: Dict) -> Dict:
    """Optimize playlist flow using Dynamic Programming"""
    n = len(tracks)
    if n == 0:
        return {"optimizedOrder": [], "flowScore": 0, "transitions": []}
    
    if n == 1:
        return {"optimizedOrder": [0], "flowScore": 1.0, "transitions": []}

    def mood_distance(m1: Dict, m2: Dict) -> float:
        v1 = m1.get('valence', 0.5)
        e1 = m1.get('energy', 0.5)
        v2 = m2.get('valence', 0.5)
        e2 = m2.get('energy', 0.5)
        d1 = m1.get('danceability', 0.5)
        d2 = m2.get('danceability', 0.5)
        
        return np.sqrt((v1 - v2)**2 + (e1 - e2)**2 + 0.3 * (d1 - d2)**2)

    track_moods = []
    for track in tracks:
        if 'moodDetails' in track and 'scores' in track['moodDetails']:
            track_moods.append(track['moodDetails']['scores'])
        elif 'mood' in track and 'scores' in track['mood']:
            track_moods.append(track['mood']['scores'])
        elif 'features' in track:
            track_moods.append(track['features'])
        else:
            track_moods.append({'valence': 0.5, 'energy': 0.5, 'danceability': 0.5})

    start_dists = np.array([mood_distance(start_mood, m) for m in track_moods])
    end_dists = np.array([mood_distance(m, end_mood) for m in track_moods])
    
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = mood_distance(track_moods[i], track_moods[j])

    dp = np.full((n, n), float('inf'))
    path = np.full((n, n), -1, dtype=int)
    
    dp[0, :] = start_dists
    
    for k in range(1, n):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cost = dp[k-1, j] + dist_matrix[j, i]
                if cost < dp[k, i]:
                    dp[k, i] = cost
                    path[k, i] = j
    
    total_costs = dp[n-1, :] + end_dists
    last_idx = np.argmin(total_costs)
    min_cost = total_costs[last_idx]
    
    ordered_indices = []
    current = last_idx
    
    for k in range(n - 1, -1, -1):
        ordered_indices.append(int(current))
        if k > 0:
            current = path[k, current]
            if current == -1:
                break
    
    ordered_indices.reverse()
    
    max_possible_cost = n * 2.0
    flow_score = max(0, 1 - (min_cost / max_possible_cost))
    
    transitions = []
    for i in range(len(ordered_indices) - 1):
        curr_idx = ordered_indices[i]
        next_idx = ordered_indices[i + 1]
        
        transition_cost = dist_matrix[curr_idx, next_idx]
        smoothness = max(0, 1 - (transition_cost / 2.0))
        
        transitions.append({
            "from_index": int(curr_idx),
            "to_index": int(next_idx),
            "smoothness": float(smoothness),
            "distance": float(transition_cost)
        })
    
    return {
        "optimizedOrder": ordered_indices,
        "flowScore": float(flow_score),
        "transitions": transitions,
        "totalCost": float(min_cost)
    }


def calculate_playlist_mood_distribution(tracks: List[Dict]) -> Dict:
    """Calculate mood distribution for playlist"""
    mood_counts = {mood: 0 for mood in ALL_MOOD_LABELS}
    
    total = len(tracks)
    if total == 0:
        return {"distribution": {}, "overall_mood": "Unknown", "total_tracks": 0}
    
    for track in tracks:
        # Handle multi-mood tracks
        track_moods = track.get('all_moods', [])
        
        if not track_moods:
            # Fallback to single mood
            mood = track.get('mood') or track.get('primary_mood')
            if isinstance(mood, dict):
                mood = mood.get('primary_mood', 'Relaxed')
            track_moods = [mood] if mood else ['Relaxed']
        
        # Count each mood (can count multiple moods per track)
        for mood in track_moods:
            if mood in mood_counts:
                mood_counts[mood] += 1
    
    distribution = {
        mood: round((count / total) * 100, 2)
        for mood, count in mood_counts.items()
        if count > 0
    }
    
    if distribution:
        overall_mood = max(distribution, key=distribution.get)
    else:
        overall_mood = "Relaxed"
    
    return {
        "distribution": distribution,
        "overall_mood": overall_mood,
        "total_tracks": total,
        "mood_diversity": len(distribution),
        "dominant_percentage": distribution.get(overall_mood, 0) if distribution else 0
    }
    
# ============================================================
# Global Model Service Wrapper for Backward Compatibility
# ============================================================

class _ModelServiceWrapper:
    async def predict_mood_from_features(self, *args, **kwargs):
        return await predict_mood_from_features(*args, **kwargs)
    
    async def predict_mood_from_spotify_track(self, *args, **kwargs):
        return await predict_mood_from_spotify_track(*args, **kwargs)
    
    async def predict_playlist_mood(self, *args, **kwargs):
        return await predict_playlist_mood(*args, **kwargs)
    
    def optimize_flow_with_gradient(self, *args, **kwargs):
        return optimize_flow_with_gradient(*args, **kwargs)
    
    def optimize_flow_dp(self, *args, **kwargs):
        return optimize_flow_dp(*args, **kwargs)
    
    def calculate_playlist_mood_distribution(self, *args, **kwargs):
        return calculate_playlist_mood_distribution(*args, **kwargs)


# Export global instance for other modules
model_service = _ModelServiceWrapper()
