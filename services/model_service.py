"""
Hybrid Model Service - FIXED VERSION
- Corrected mood classification logic
- Proper feature extraction and normalization
- Genre-adaptive weighting with correct mood mapping
- FIXED: dtype conversion for ONNX (float32)
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
            return metadata.get('mood_classes', ["Calm", "Energetic", "Happy", "Sad"])
    return ["Calm", "Energetic", "Happy", "Sad"]

MOOD_CLASSES = load_mood_classes()
print(f"🎭 Loaded mood classes: {MOOD_CLASSES}")

MODEL_FEATURE_ORDER = [
    'valence', 'energy', 'danceability', 'acousticness', 
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

# Genre weights for lyrics vs audio (CORRECTED)
GENRE_WEIGHTS = {
    'pop': {'audio': 0.6, 'lyrics': 0.4},
    'rock': {'audio': 0.7, 'lyrics': 0.3},
    'hip-hop': {'audio': 0.4, 'lyrics': 0.6},
    'rap': {'audio': 0.35, 'lyrics': 0.65},  # Rap is MORE lyrics-heavy
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
                scaler_mean = np.array(metadata['scaler_mean'], dtype=np.float32)  # FIX: Ensure float32
                scaler_scale = np.array(metadata['scaler_scale'], dtype=np.float32)  # FIX: Ensure float32
                MOOD_CLASSES = metadata['mood_classes']
                print(f"✅ Metadata loaded: {MOOD_CLASSES}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}. Using rule-based fallback.")
        mood_model = None


def normalize_features(features: Dict) -> np.ndarray:
    """
    Normalize audio features using saved scaler
    FIX: Ensures proper float32 dtype for ONNX
    """
    feature_values = []
    for feature_name in MODEL_FEATURE_ORDER:
        value = features.get(feature_name, 0.5)
        feature_values.append(float(value))
    
    # FIX: Create array as float32 from the start
    feature_array = np.array(feature_values, dtype=np.float32)
    
    # Apply standardization if scaler available
    if scaler_mean is not None and scaler_scale is not None:
        feature_array = (feature_array - scaler_mean) / scaler_scale
        # FIX: Ensure result is still float32 (not float64)
        feature_array = feature_array.astype(np.float32)
    
    return feature_array


async def predict_mood_from_spotify_track(
    track_id: str,
    access_token: str,
    lyrics_sentiment: Dict,
    user_id: Optional[str] = None
) -> Dict:
    """
    HYBRID APPROACH: Predict mood for Spotify track
    """
    # Check user override first
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
    
    # Get track metadata from Spotify
    track_info = await sp_service.get_track_info(track_id, access_token)
    
    if not track_info:
        raise ValueError(f"Track {track_id} not found on Spotify")
    
    track_name = track_info['name']
    artist_name = sp_service.get_primary_artist_name(track_info)
    
    print(f"🎵 Analyzing: {track_name} by {artist_name}")
    
    # FIX: Try Spotify audio features first (MOST RELIABLE)
    audio_features = None
    
    # Check if we have Spotify client credentials
    if os.getenv("SPOTIFY_CLIENT_ID") and os.getenv("SPOTIFY_CLIENT_SECRET"):
        try:
            # Get Spotify's native audio features
            sp_client = sp_service.get_spotify_client(access_token)
            spotify_features = sp_client.audio_features([track_id])
            
            if spotify_features and spotify_features[0]:
                audio_features = spotify_features[0]
                print(f"✅ Using Spotify audio features (BEST)")
        except Exception as e:
            print(f"⚠️ Spotify audio features failed: {e}")
    
    # Fallback to Multi-API if Spotify features unavailable
    if not audio_features:
        print(f"⚠️ Falling back to multi-API feature extraction")
        audio_features = await music_service.get_audio_features(
            track_name,
            artist_name
        )
    
    if not audio_features:
        print("⚠️ Using default audio features")
        audio_features = music_service.get_default_features()
    
    # Get genre tags from Last.fm
    tags = await music_service.get_lastfm_tags(track_name, artist_name)
    genre = tags[0].lower() if tags else None
    
    print(f"🎸 Genre detected: {genre}")
    
    # Predict mood
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
    Core mood prediction logic - FIXED VERSION
    """
    
    audio_mood = "Calm"  # Default
    confidence = 0.0
    mood_probabilities = None
    
    valence = float(audio_features.get('valence', 0.5))
    energy = float(audio_features.get('energy', 0.5))
    danceability = float(audio_features.get('danceability', 0.5))
    
    print(f"📊 Audio features - Valence: {valence:.2f}, Energy: {energy:.2f}, Dance: {danceability:.2f}")

    # 1. Audio-based mood using ONNX model
    if mood_model:
        try:
            input_data = normalize_features(audio_features)
            # FIX: Ensure reshape maintains float32
            input_data = input_data.reshape(1, -1).astype(np.float32)
            
            input_name = mood_model.get_inputs()[0].name
            nn_output = mood_model.run(None, {input_name: input_data})[0]
            mood_probabilities = nn_output[0]
            
            # Apply personalized adjustments if user exists
            if user_id:
                mood_probabilities = await apply_personalized_adjustments(
                    audio_features,
                    mood_probabilities,
                    user_id
                )
            
            predicted_index = np.argmax(mood_probabilities)
            confidence = float(mood_probabilities[predicted_index])
            audio_mood = MOOD_CLASSES[predicted_index]
            
            print(f"🤖 Model prediction: {audio_mood} ({confidence:.2%})")
            print(f"   All probabilities: {dict(zip(MOOD_CLASSES, mood_probabilities))}")

        except Exception as e:
            print(f"⚠️ Model prediction failed: {e}")
            audio_mood, confidence = _rule_based_mood(valence, energy, danceability)
            
    else:
        audio_mood, confidence = _rule_based_mood(valence, energy, danceability)
    
    # 2. Lyrics sentiment analysis
    lyric_polarity = float(lyrics_sentiment.get('polarity', 0.0))
    lyric_subjectivity = float(lyrics_sentiment.get('subjectivity', 0.0))
    
    print(f"📝 Lyrics - Polarity: {lyric_polarity:.2f}, Subjectivity: {lyric_subjectivity:.2f}")

    # 3. Genre-adaptive fusion
    weights = GENRE_WEIGHTS.get(genre if genre else 'default', GENRE_WEIGHTS['default'])
    
    # Calculate lyrics strength
    lyric_strength = abs(lyric_polarity) * lyric_subjectivity
    
    # Adjust weights based on lyrics strength
    if lyric_strength > 0.5:
        # Strong lyrics - increase lyrics weight
        lyric_weight = min(weights['lyrics'] * (1 + lyric_strength), 0.8)
    else:
        # Weak lyrics - decrease lyrics weight
        lyric_weight = weights['lyrics'] * lyric_strength
    
    audio_weight = 1.0 - lyric_weight
    
    print(f"⚖️ Fusion weights - Audio: {audio_weight:.2f}, Lyrics: {lyric_weight:.2f}")
    
    # 4. Determine lyrics mood influence
    if lyric_polarity > 0.3:
        lyrics_mood = "Positive"
        lyric_valence_boost = 0.2
    elif lyric_polarity < -0.3:
        lyrics_mood = "Negative"
        lyric_valence_boost = -0.2
    else:
        lyrics_mood = "Neutral"
        lyric_valence_boost = 0.0
    
    # 5. Apply fusion
    # Adjust valence based on lyrics
    fused_valence = valence + (lyric_valence_boost * lyric_weight)
    fused_valence = max(0.0, min(1.0, fused_valence))  # Clamp to [0, 1]
    
    # Keep energy from audio (lyrics don't affect energy much)
    fused_energy = energy
    
    # Classify final mood
    fused_mood = _classify_mood_from_valence_energy(fused_valence, fused_energy, danceability)
    
    print(f"🎯 Final mood: {fused_mood} (Audio: {audio_mood}, Lyrics: {lyrics_mood})")
    
    return {
        "audio_mood": audio_mood,
        "lyrics_mood": lyrics_mood,
        "fused_mood": fused_mood,
        "confidence": confidence,
        "source": "ml_model_personalized" if (mood_model and user_id) else ("ml_model" if mood_model else "rule_based"),
        "scores": {
            "valence": valence,
            "energy": energy,
            "danceability": danceability,
            "acousticness": float(audio_features.get('acousticness', 0.5)),
            "lyrics_polarity": lyric_polarity,
            "lyrics_subjectivity": lyric_subjectivity,
            "fused_valence": fused_valence,
            "fused_energy": fused_energy,
            "audio_weight": audio_weight,
            "lyric_weight": lyric_weight
        },
        "genre": genre,
        "model_probabilities": dict(zip(MOOD_CLASSES, mood_probabilities)) if mood_probabilities is not None else None
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
    
    # Renormalize
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    return adjusted_probs


def _rule_based_mood(valence: float, energy: float, danceability: float) -> tuple:
    """
    Rule-based fallback with improved logic
    """
    # High energy, high valence, high danceability = Happy
    if valence > 0.6 and energy > 0.6:
        return "Happy", 0.75
    
    # High energy, any valence = Energetic
    elif energy > 0.7:
        return "Energetic", 0.70
    
    # Low energy, high valence = Calm
    elif energy < 0.4 and valence > 0.4:
        return "Calm", 0.70
    
    # Low energy, low valence = Sad
    elif energy < 0.4 and valence < 0.4:
        return "Sad", 0.75
    
    # Medium energy, low valence = Sad
    elif valence < 0.3:
        return "Sad", 0.65
    
    # Default to Calm for ambiguous cases
    else:
        return "Calm", 0.50


def _classify_mood_from_valence_energy(valence: float, energy: float, danceability: float = 0.5) -> str:
    """
    Classify mood from valence, energy, and danceability
    IMPROVED LOGIC based on music psychology
    """
    
    # Very high energy (> 0.75) = Energetic (regardless of valence)
    if energy > 0.75:
        return "Energetic"
    
    # High energy + High valence = Happy
    if energy > 0.6 and valence > 0.6:
        return "Happy"
    
    # High energy + Low valence = Energetic (angry/intense music)
    if energy > 0.6 and valence < 0.4:
        return "Energetic"
    
    # Low energy + Low valence = Sad
    if energy < 0.4 and valence < 0.4:
        return "Sad"
    
    # Low energy + High valence = Calm
    if energy < 0.4 and valence > 0.5:
        return "Calm"
    
    # Medium energy + High valence = Happy
    if valence > 0.65:
        return "Happy"
    
    # Low valence = Sad
    if valence < 0.35:
        return "Sad"
    
    # Default: Calm (for neutral/ambiguous cases)
    return "Calm"


def map_external_mood_to_base(external_mood: str) -> str:
    """
    Map any external mood to one of the 4 base moods
    This allows handling of moods not in training data
    """
    external_mood_lower = external_mood.lower().strip()
    
    # Mapping dictionary
    mood_mapping = {
        # Happy variations
        'joyful': 'Happy',
        'cheerful': 'Happy',
        'upbeat': 'Happy',
        'excited': 'Happy',
        'ecstatic': 'Happy',
        'delighted': 'Happy',
        'pleased': 'Happy',
        'content': 'Happy',
        
        # Sad variations
        'melancholic': 'Sad',
        'depressed': 'Sad',
        'gloomy': 'Sad',
        'sorrowful': 'Sad',
        'unhappy': 'Sad',
        'blue': 'Sad',
        'down': 'Sad',
        'lonely': 'Sad',
        
        # Calm variations
        'relaxed': 'Calm',
        'peaceful': 'Calm',
        'chill': 'Calm',
        'serene': 'Calm',
        'tranquil': 'Calm',
        'mellow': 'Calm',
        'laid-back': 'Calm',
        'soothing': 'Calm',
        'ambient': 'Calm',
        
        # Energetic variations
        'hyped': 'Energetic',
        'pumped': 'Energetic',
        'intense': 'Energetic',
        'powerful': 'Energetic',
        'aggressive': 'Energetic',
        'angry': 'Energetic',
        'fierce': 'Energetic',
        'dynamic': 'Energetic',
        'vigorous': 'Energetic',
        
        # Activity-based moods
        'workout': 'Energetic',
        'gym': 'Energetic',
        'party': 'Happy',
        'study': 'Calm',
        'focus': 'Calm',
        'sleep': 'Calm',
        'meditation': 'Calm',
        'driving': 'Energetic',
        'romantic': 'Calm',
        'dance': 'Energetic'
    }
    
    # Check direct mapping
    if external_mood_lower in mood_mapping:
        return mood_mapping[external_mood_lower]
    
    # Check if already a base mood
    for base_mood in MOOD_CLASSES:
        if external_mood_lower == base_mood.lower():
            return base_mood
    
    # Algorithmic determination based on keyword matching
    if any(word in external_mood_lower for word in ['happy', 'joy', 'cheer', 'up', 'bright']):
        return 'Happy'
    elif any(word in external_mood_lower for word in ['sad', 'depress', 'melan', 'sorrow', 'blue']):
        return 'Sad'
    elif any(word in external_mood_lower for word in ['calm', 'relax', 'peace', 'chill', 'sooth']):
        return 'Calm'
    elif any(word in external_mood_lower for word in ['energy', 'hype', 'intense', 'power', 'aggressive']):
        return 'Energetic'
    
    # Default to Calm for unknown moods
    print(f"⚠️ Unknown mood '{external_mood}', defaulting to Calm")
    return 'Calm'


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
            # Map to base mood if necessary
            mood = map_external_mood_to_base(mood)
        else:
            mood = 'Calm'
        
        if mood in mood_counts:
            mood_counts[mood] += 1
        else:
            # Try to map unknown mood
            mapped_mood = map_external_mood_to_base(mood)
            if mapped_mood in mood_counts:
                mood_counts[mapped_mood] += 1
    
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