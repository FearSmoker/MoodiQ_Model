"""
Hybrid Model Service
- Gets track metadata from Spotify
- Gets audio features from Multi-API (MusicBrainz → AcousticBrainz)
- Predicts mood using ML model + lyrics sentiment
"""

import numpy as np
import onnxruntime as ort
import os
import json
from typing import Dict, List, Optional
from . import cache_service
from . import music_service
from . import spotify_service as sp_service

# Model Configuration
MOOD_MODEL_PATH = os.path.join("models", "mood_model.onnx")

def load_mood_classes():
    """Load mood classes from metadata"""
    metadata_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('mood_classes', ["Happy", "Sad", "Calm", "Energetic"])
    return ["Happy", "Sad", "Calm", "Energetic"]

MOOD_CLASSES = load_mood_classes()

MODEL_FEATURE_ORDER = [
    'valence', 'energy', 'danceability', 'acousticness', 
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

GENRE_WEIGHTS = {
    'pop': {'audio': 0.6, 'lyrics': 0.4},
    'rock': {'audio': 0.7, 'lyrics': 0.3},
    'hip-hop': {'audio': 0.4, 'lyrics': 0.6},
    'rap': {'audio': 0.3, 'lyrics': 0.7},
    'electronic': {'audio': 0.8, 'lyrics': 0.2},
    'edm': {'audio': 0.8, 'lyrics': 0.2},
    'classical': {'audio': 0.9, 'lyrics': 0.1},
    'indie': {'audio': 0.5, 'lyrics': 0.5},
    'r&b': {'audio': 0.5, 'lyrics': 0.5},
    'rnb': {'audio': 0.5, 'lyrics': 0.5},
    'country': {'audio': 0.5, 'lyrics': 0.5},
    'jazz': {'audio': 0.8, 'lyrics': 0.2},
    'default': {'audio': 0.6, 'lyrics': 0.4}
}

MOOD_MAPPING = {
    'Angry': 'Energetic',
    'Focus': 'Calm',
    'Neutral': 'Calm'
}

# Model globals
mood_model = None
session_options = None
scaler_mean = None
scaler_scale = None


def load_model():
    """Load ONNX model"""
    global mood_model, session_options, scaler_mean, scaler_scale, MOOD_CLASSES
    
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
                scaler_mean = np.array(metadata['scaler_mean'])
                scaler_scale = np.array(metadata['scaler_scale'])
                MOOD_CLASSES = metadata['mood_classes']
                print(f"✅ Metadata loaded: {MOOD_CLASSES}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}. Using rule-based fallback.")
        mood_model = None


def normalize_features(features: Dict) -> np.ndarray:
    """Normalize audio features"""
    feature_values = []
    for feature_name in MODEL_FEATURE_ORDER:
        value = features.get(feature_name, 0.5)
        feature_values.append(value)
    
    feature_array = np.array(feature_values, dtype=np.float32)
    
    if scaler_mean is not None and scaler_scale is not None:
        feature_array = (feature_array - scaler_mean) / scaler_scale
    else:
        tempo_idx = MODEL_FEATURE_ORDER.index('tempo')
        loudness_idx = MODEL_FEATURE_ORDER.index('loudness')
        key_idx = MODEL_FEATURE_ORDER.index('key')
        time_sig_idx = MODEL_FEATURE_ORDER.index('time_signature')
        
        feature_array[tempo_idx] = min(feature_array[tempo_idx] / 200.0, 1.0)
        feature_array[loudness_idx] = (feature_array[loudness_idx] + 60) / 60.0
        feature_array[key_idx] = feature_array[key_idx] / 11.0
        feature_array[time_sig_idx] = feature_array[time_sig_idx] / 7.0
    
    return feature_array


async def predict_mood_from_spotify_track(
    track_id: str,
    access_token: str,
    lyrics_sentiment: Dict,
    user_id: Optional[str] = None
) -> Dict:
    """
    HYBRID APPROACH: Predict mood for Spotify track
    
    Pipeline:
    1. Get track metadata from Spotify
    2. Extract artist & track names
    3. Get audio features from Multi-API (MusicBrainz → AcousticBrainz)
    4. Get genre from Spotify artist data
    5. Predict mood with ML model
    
    Args:
        track_id: Spotify track ID
        access_token: User's Spotify access token
        lyrics_sentiment: Lyrics sentiment dict
        user_id: Optional user ID for personalization
    """
    # Check user override
    if user_id and track_id:
        override_key = f"user_model:{user_id}:track:{track_id}"
        cached_override = await cache_service.get_from_cache(override_key)
        if cached_override:
            return {
                "track_id": track_id,
                "fused_mood": cached_override,
                "confidence": 1.0,
                "source": "user_feedback"
            }
    
    # 1. Get track metadata from Spotify
    track_info = await sp_service.get_track_info(track_id, access_token)
    
    if not track_info:
        raise ValueError(f"Track {track_id} not found on Spotify")
    
    track_name = track_info['name']
    artist_name = sp_service.get_primary_artist_name(track_info)
    
    print(f"🎵 Analyzing: {track_name} by {artist_name}")
    
    # 2. Get audio features from Multi-API
    audio_features = await music_service.get_audio_features(
        track_name,
        artist_name
    )
    
    if not audio_features:
        print("⚠️ Using default audio features")
        audio_features = music_service.get_default_features()
    
    # 3. Get genre from Spotify (for adaptive weighting)
    genre = None
    if track_info.get('artists'):
        # Get first artist's genres
        artist_id = track_info['artists'][0]['id']
        # Note: Would need additional API call to get artist genres
        # For now, use Last.fm tags as fallback
        tags = await music_service.get_lastfm_tags(track_name, artist_name)
        genre = tags[0] if tags else None
    
    # 4. Predict mood
    mood_data = await predict_mood_from_features(
        audio_features,
        lyrics_sentiment,
        user_id=user_id,
        track_id=track_id,
        genre=genre
    )
    
    # Add Spotify metadata to response
    mood_data['track_info'] = {
        'id': track_info['id'],
        'name': track_info['name'],
        'artists': [a['name'] for a in track_info['artists']],
        'album': track_info['album']['name'],
        'popularity': track_info['popularity'],
        'duration_ms': track_info['duration_ms'],
        'external_url': track_info['external_url']
    }
    
    return mood_data


async def predict_mood_from_features(
    audio_features: Dict, 
    lyrics_sentiment: Dict, 
    user_id: Optional[str] = None,
    track_id: Optional[str] = None,
    genre: Optional[str] = None
) -> Dict:
    """Core mood prediction logic"""
    
    audio_mood = "Neutral"
    confidence = 0.0
    valence = audio_features.get('valence', 0.5)
    energy = audio_features.get('energy', 0.5)

    # 1. Audio-based mood using ONNX model
    if mood_model:
        try:
            input_data = normalize_features(audio_features)
            input_data = input_data.reshape(1, -1)
            
            input_name = mood_model.get_inputs()[0].name
            nn_output = mood_model.run(None, {input_name: input_data})[0]
            mood_probabilities = nn_output[0]
            
            if user_id:
                mood_probabilities = await apply_personalized_adjustments(
                    audio_features,
                    mood_probabilities,
                    user_id
                )
            
            predicted_index = np.argmax(mood_probabilities)
            confidence = float(mood_probabilities[predicted_index])
            audio_mood = MOOD_CLASSES[predicted_index]
            
            print(f"🤖 Model: {audio_mood} ({confidence:.2f})")

        except Exception as e:
            print(f"⚠️ Model prediction failed: {e}")
            audio_mood, confidence = _rule_based_mood(valence, energy)
            
    else:
        audio_mood, confidence = _rule_based_mood(valence, energy)
    
    # 2. Lyrics sentiment
    lyric_polarity = lyrics_sentiment.get('polarity', 0.0)
    lyric_subjectivity = lyrics_sentiment.get('subjectivity', 0.0)

    # 3. Adaptive fusion
    weights = GENRE_WEIGHTS.get(genre.lower() if genre else 'default', GENRE_WEIGHTS['default'])
    
    lyric_strength = abs(lyric_polarity) * lyric_subjectivity
    
    if lyric_strength > 0.5:
        lyric_weight = min(weights['lyrics'] * (1 + lyric_strength), 0.8)
    else:
        lyric_weight = weights['lyrics'] * lyric_strength
    
    audio_weight = 1.0 - lyric_weight
    
    lyric_valence_equivalent = (lyric_polarity + 1) / 2
    final_valence = (valence * audio_weight) + (lyric_valence_equivalent * lyric_weight)
    
    fused_mood = _classify_mood_from_valence_energy(final_valence, energy)
    
    if lyric_polarity > 0.2:
        lyrics_mood = "Positive"
    elif lyric_polarity < -0.2:
        lyrics_mood = "Negative"
    else:
        lyrics_mood = "Neutral"
        
    return {
        "audio_mood": audio_mood,
        "lyrics_mood": lyrics_mood,
        "fused_mood": fused_mood,
        "confidence": confidence,
        "source": "ml_model_personalized" if (mood_model and user_id) else ("ml_model" if mood_model else "rule_based"),
        "scores": {
            "valence": float(valence),
            "energy": float(energy),
            "danceability": float(audio_features.get('danceability', 0.5)),
            "acousticness": float(audio_features.get('acousticness', 0.5)),
            "lyrics_polarity": float(lyric_polarity),
            "lyrics_subjectivity": float(lyric_subjectivity),
            "fused_valence": float(final_valence),
            "audio_weight": float(audio_weight),
            "lyric_weight": float(lyric_weight)
        }
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
    
    for i, mood in enumerate(MOOD_CLASSES):
        if mood in mood_weights:
            boost = mood_weights[mood] * 0.3
            adjusted_probs[i] = adjusted_probs[i] * (1 + boost)
    
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    return adjusted_probs


def _rule_based_mood(valence: float, energy: float) -> tuple:
    """Rule-based fallback"""
    if valence > 0.6 and energy > 0.6:
        return "Happy", 0.7
    elif valence > 0.6 and energy <= 0.4:
        return "Calm", 0.7
    elif valence <= 0.4 and energy > 0.6:
        return "Energetic", 0.7
    elif valence <= 0.4 and energy <= 0.4:
        return "Sad", 0.7
    elif energy > 0.7:
        return "Energetic", 0.6
    else:
        return "Calm", 0.5


def _classify_mood_from_valence_energy(valence: float, energy: float) -> str:
    """Classify mood from valence and energy"""
    if valence > 0.6 and energy > 0.6:
        return "Happy"
    elif valence > 0.6 and energy <= 0.4:
        return "Calm"
    elif valence <= 0.4 and energy > 0.6:
        return "Energetic"
    elif valence <= 0.4 and energy <= 0.4:
        return "Sad"
    elif energy > 0.7:
        return "Energetic"
    else:
        return "Calm"


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
        if 'mood' in track and 'scores' in track['mood']:
            track_moods.append(track['mood']['scores'])
        elif 'moodDetails' in track and 'scores' in track['moodDetails']:
            track_moods.append(track['moodDetails']['scores'])
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


def detect_mood_gaps(tracks: List[Dict], threshold: float = 1.5) -> List[Dict]:
    """Detect mood gaps in playlist"""
    gaps = []
    
    for i in range(len(tracks) - 1):
        current_mood = tracks[i].get('moodDetails', {}).get('scores', {}) or tracks[i].get('mood', {}).get('scores', {})
        next_mood = tracks[i + 1].get('moodDetails', {}).get('scores', {}) or tracks[i + 1].get('mood', {}).get('scores', {})
        
        v1 = current_mood.get('valence', 0.5)
        e1 = current_mood.get('energy', 0.5)
        v2 = next_mood.get('valence', 0.5)
        e2 = next_mood.get('energy', 0.5)
        
        distance = np.sqrt((v1 - v2)**2 + (e1 - e2)**2)
        
        if distance > threshold:
            bridge_valence = (v1 + v2) / 2
            bridge_energy = (e1 + e2) / 2
            
            gaps.append({
                "position": i + 1,
                "from_track": tracks[i].get('name', 'Unknown'),
                "to_track": tracks[i + 1].get('name', 'Unknown'),
                "distance": float(distance),
                "severity": "high" if distance > 2.0 else "medium",
                "recommended_bridge_mood": {
                    "valence": float(bridge_valence),
                    "energy": float(bridge_energy)
                }
            })
    
    return gaps


def calculate_playlist_mood_distribution(tracks: List[Dict]) -> Dict:
    """Calculate mood distribution"""
    mood_counts = {mood: 0 for mood in MOOD_CLASSES}
    
    total = len(tracks)
    if total == 0:
        return {"distribution": {}, "overall_mood": "Unknown", "total_tracks": 0}
    
    for track in tracks:
        mood = track.get('mood')
        if isinstance(mood, dict):
            mood = mood.get('fused_mood', 'Calm')
        elif isinstance(mood, str):
            pass
        else:
            mood = 'Calm'
        
        if mood in MOOD_MAPPING:
            mood = MOOD_MAPPING[mood]
        
        if mood in mood_counts:
            mood_counts[mood] += 1
        else:
            mood_counts['Calm'] += 1
    
    distribution = {
        mood: round((count / total) * 100, 2)
        for mood, count in mood_counts.items()
        if count > 0
    }
    
    if distribution:
        overall_mood = max(distribution, key=distribution.get)
    else:
        overall_mood = "Calm"
    
    return {
        "distribution": distribution,
        "overall_mood": overall_mood,
        "total_tracks": total,
        "mood_diversity": len(distribution),
        "dominant_percentage": distribution.get(overall_mood, 0) if distribution else 0
    }