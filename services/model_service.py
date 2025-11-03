import numpy as np
import onnxruntime as ort
import os
import json
from typing import Dict, List, Optional
from . import cache_service

# --- Model Configuration ---
MOOD_MODEL_PATH = os.path.join("models", "mood_model.onnx")

# Load mood classes from metadata (dynamic based on training)
def load_mood_classes():
    """Load mood classes from model metadata"""
    metadata_path = os.path.join("models", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('mood_classes', ["Happy", "Sad", "Calm", "Energetic"])
    return ["Happy", "Sad", "Calm", "Energetic"]

# Dynamically load mood classes
MOOD_CLASSES = load_mood_classes()

# Define the order of features your model expects
MODEL_FEATURE_ORDER = [
    'valence', 'energy', 'danceability', 'acousticness', 
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

# Genre-specific weights for adaptive fusion
GENRE_WEIGHTS = {
    'pop': {'audio': 0.6, 'lyrics': 0.4},
    'rock': {'audio': 0.7, 'lyrics': 0.3},
    'hip-hop': {'audio': 0.4, 'lyrics': 0.6},
    'rap': {'audio': 0.3, 'lyrics': 0.7},
    'electronic': {'audio': 0.8, 'lyrics': 0.2},
    'classical': {'audio': 0.9, 'lyrics': 0.1},
    'indie': {'audio': 0.5, 'lyrics': 0.5},
    'r&b': {'audio': 0.5, 'lyrics': 0.5},
    'country': {'audio': 0.5, 'lyrics': 0.5},
    'jazz': {'audio': 0.8, 'lyrics': 0.2},
    'default': {'audio': 0.6, 'lyrics': 0.4}
}

# Extended mood mapping (maps similar moods to trained classes)
MOOD_MAPPING = {
    'Angry': 'Energetic',  # Map Angry to Energetic (similar energy profile)
    'Focus': 'Calm',       # Map Focus to Calm (similar concentration profile)
    'Neutral': 'Calm'      # Map Neutral to Calm
}

# Placeholder for the loaded model and scaler
mood_model = None
session_options = None
scaler_mean = None
scaler_scale = None


def load_model():
    """
    Loads the trained ONNX machine learning model and metadata from disk.
    """
    global mood_model, session_options, scaler_mean, scaler_scale, MOOD_CLASSES
    
    try:
        # Load model
        if not os.path.exists(MOOD_MODEL_PATH):
            print(f"⚠️  WARNING: Model file not found at {MOOD_MODEL_PATH}. Using rule-based fallback.")
            return
        
        # Configure ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load ONNX model
        mood_model = ort.InferenceSession(
            MOOD_MODEL_PATH, 
            session_options,
            providers=['CPUExecutionProvider']
        )
        
        print(f"✅ ONNX model loaded successfully from {MOOD_MODEL_PATH}")
        
        # Print model info
        input_info = mood_model.get_inputs()[0]
        output_info = mood_model.get_outputs()[0]
        print(f"   Input: {input_info.name}, Shape: {input_info.shape}, Type: {input_info.type}")
        print(f"   Output: {output_info.name}, Shape: {output_info.shape}, Type: {output_info.type}")
        
        # Load metadata for normalization
        metadata_path = os.path.join("models", "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                scaler_mean = np.array(metadata['scaler_mean'])
                scaler_scale = np.array(metadata['scaler_scale'])
                MOOD_CLASSES = metadata['mood_classes']
                print(f"✅ Loaded metadata: {len(MOOD_CLASSES)} mood classes")
                print(f"   Mood classes: {MOOD_CLASSES}")
        else:
            print(f"⚠️  WARNING: Metadata file not found. Using default normalization.")
        
    except Exception as e:
        print(f"❌ ERROR: Could not load ONNX model: {e}. Using rule-based fallback.")
        mood_model = None


def normalize_features(features: Dict) -> np.ndarray:
    """
    Normalize audio features using saved scaler parameters.
    """
    # Extract features in correct order
    feature_values = []
    for feature_name in MODEL_FEATURE_ORDER:
        value = features.get(feature_name, 0.5)
        feature_values.append(value)
    
    feature_array = np.array(feature_values, dtype=np.float32)
    
    # Apply standardization if scaler is available
    if scaler_mean is not None and scaler_scale is not None:
        feature_array = (feature_array - scaler_mean) / scaler_scale
    else:
        # Manual normalization for tempo and loudness
        tempo_idx = MODEL_FEATURE_ORDER.index('tempo')
        loudness_idx = MODEL_FEATURE_ORDER.index('loudness')
        key_idx = MODEL_FEATURE_ORDER.index('key')
        time_sig_idx = MODEL_FEATURE_ORDER.index('time_signature')
        
        feature_array[tempo_idx] = min(feature_array[tempo_idx] / 200.0, 1.0)
        feature_array[loudness_idx] = (feature_array[loudness_idx] + 60) / 60.0
        feature_array[key_idx] = feature_array[key_idx] / 11.0
        feature_array[time_sig_idx] = feature_array[time_sig_idx] / 7.0
    
    return feature_array


async def apply_personalized_adjustments(
    features: Dict,
    mood_probabilities: np.ndarray,
    user_id: Optional[str]
) -> np.ndarray:
    """
    Apply user-specific model adjustments to mood predictions.
    This implements the personalization layer.
    """
    if not user_id:
        return mood_probabilities
    
    # Get user's trained model
    user_model_key = f"user_model:{user_id}:trained"
    user_model = await cache_service.get_from_cache(user_model_key)
    
    if not user_model:
        return mood_probabilities
    
    print(f"🎯 Applying personalized adjustments for user {user_id}")
    
    # Get mood weights and feature adjustments
    mood_weights = user_model.get('mood_weights', {})
    feature_adjustments = user_model.get('feature_adjustments', {})
    
    # Apply feature-based adjustments
    valence = features.get('valence', 0.5)
    energy = features.get('energy', 0.5)
    
    # Adjust valence and energy based on learned biases
    adjusted_valence = valence + feature_adjustments.get('valence_bias', 0.0)
    adjusted_energy = energy + feature_adjustments.get('energy_bias', 0.0)
    
    # Clip to valid range
    adjusted_valence = np.clip(adjusted_valence, 0, 1)
    adjusted_energy = np.clip(adjusted_energy, 0, 1)
    
    # Recalculate mood distribution with adjustments
    adjusted_probs = mood_probabilities.copy()
    
    # Boost probabilities for user's preferred moods
    for i, mood in enumerate(MOOD_CLASSES):
        if mood in mood_weights:
            boost = mood_weights[mood] * 0.3  # Up to 30% boost
            adjusted_probs[i] = adjusted_probs[i] * (1 + boost)
    
    # Re-normalize to sum to 1
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    
    print(f"   Original top mood: {MOOD_CLASSES[np.argmax(mood_probabilities)]}")
    print(f"   Adjusted top mood: {MOOD_CLASSES[np.argmax(adjusted_probs)]}")
    
    return adjusted_probs


async def predict_mood_from_features(
    audio_features: Dict, 
    lyrics_sentiment: Dict, 
    user_id: Optional[str] = None,
    track_id: Optional[str] = None,
    genre: Optional[str] = None
) -> Dict:
    """
    Combines audio features and lyrics sentiment to predict a fused mood.
    Implements user-specific overrides and adaptive weighting with personalization.
    """
    
    # Check for user-specific override first (personalized learning)
    if user_id and track_id:
        override_key = f"user_model:{user_id}:track:{track_id}"
        cached_override = await cache_service.get_from_cache(override_key)
        if cached_override:
            print(f"🎯 Using user override for track {track_id}: {cached_override}")
            return {
                "audio_mood": cached_override,
                "lyrics_mood": "User Override",
                "fused_mood": cached_override,
                "confidence": 1.0,
                "source": "user_feedback",
                "scores": {
                    "valence": audio_features.get('valence', 0.5),
                    "energy": audio_features.get('energy', 0.5),
                    "lyrics_polarity": lyrics_sentiment.get('polarity', 0.0),
                }
            }
    
    audio_mood = "Neutral"
    confidence = 0.0
    valence = audio_features.get('valence', 0.5)
    energy = audio_features.get('energy', 0.5)

    # 1. Get Audio-based mood using ONNX model
    if mood_model:
        try:
            # Normalize features
            input_data = normalize_features(audio_features)
            input_data = input_data.reshape(1, -1)
            
            # Get model input name
            input_name = mood_model.get_inputs()[0].name
            
            # Run inference
            nn_output = mood_model.run(None, {input_name: input_data})[0]
            mood_probabilities = nn_output[0]
            
            # Apply personalized adjustments if user model exists
            if user_id:
                mood_probabilities = await apply_personalized_adjustments(
                    audio_features,
                    mood_probabilities,
                    user_id
                )
            
            # Get prediction and confidence
            predicted_index = np.argmax(mood_probabilities)
            confidence = float(mood_probabilities[predicted_index])
            audio_mood = MOOD_CLASSES[predicted_index]
            
            print(f"🤖 Model Prediction: {audio_mood} (confidence: {confidence:.2f})")
            if user_id:
                print(f"   (Personalized for user {user_id})")

        except Exception as e:
            print(f"⚠️  NN model prediction failed: {e}. Falling back to rules.")
            import traceback
            traceback.print_exc()
            audio_mood, confidence = _rule_based_mood(valence, energy)
            
    else:
        # Rule-based quadrant mapping (fallback if model isn't loaded)
        audio_mood, confidence = _rule_based_mood(valence, energy)
    
    # 2. Lyrics-based sentiment
    lyric_polarity = lyrics_sentiment.get('polarity', 0.0)
    lyric_subjectivity = lyrics_sentiment.get('subjectivity', 0.0)

    # 3. Adaptive Lyrics Fusion Weighting
    # Get genre-specific weights if available
    weights = GENRE_WEIGHTS.get(genre.lower() if genre else 'default', GENRE_WEIGHTS['default'])
    
    # Adjust based on lyric strength (stronger sentiment = more weight)
    lyric_strength = abs(lyric_polarity) * lyric_subjectivity
    
    # Increase lyric weight if lyrics are strongly polarized and subjective
    if lyric_strength > 0.5:
        lyric_weight = min(weights['lyrics'] * (1 + lyric_strength), 0.8)
    else:
        lyric_weight = weights['lyrics'] * lyric_strength
    
    audio_weight = 1.0 - lyric_weight
    
    print(f"⚖️  Fusion weights - Audio: {audio_weight:.2f}, Lyrics: {lyric_weight:.2f}")
    
    # Normalize lyric_polarity to be [0, 1] like valence
    lyric_valence_equivalent = (lyric_polarity + 1) / 2
    
    # Fused valence
    final_valence = (valence * audio_weight) + (lyric_valence_equivalent * lyric_weight)
    
    # Re-classify mood based on fused valence and original energy
    fused_mood = _classify_mood_from_valence_energy(final_valence, energy)
    
    # Determine lyrics mood
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


def _rule_based_mood(valence: float, energy: float) -> tuple:
    """
    Simple rule-based mood classification based on Russell's circumplex model.
    Returns (mood, confidence)
    """
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
    """
    Classify mood from valence and energy using circumplex model.
    Maps to trained mood classes.
    """
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
    """
    Optimizes the playlist flow using Dynamic Programming to find the
    smoothest path through the mood space.
    
    Returns optimized order with transition scores.
    """
    print(f"🔄 Optimizing flow using Dynamic Programming...")
    
    n = len(tracks)
    if n == 0:
        return {"optimizedOrder": [], "flowScore": 0, "transitions": []}
    
    if n == 1:
        return {
            "optimizedOrder": [0],
            "flowScore": 1.0,
            "transitions": []
        }

    def mood_distance(m1: Dict, m2: Dict) -> float:
        """Calculate Euclidean distance in mood space."""
        v1 = m1.get('valence', 0.5)
        e1 = m1.get('energy', 0.5)
        v2 = m2.get('valence', 0.5)
        e2 = m2.get('energy', 0.5)
        
        # Include danceability for better transitions
        d1 = m1.get('danceability', 0.5)
        d2 = m2.get('danceability', 0.5)
        
        return np.sqrt((v1 - v2)**2 + (e1 - e2)**2 + 0.3 * (d1 - d2)**2)

    # Extract mood scores from tracks
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

    # Build cost matrices
    start_dists = np.array([mood_distance(start_mood, m) for m in track_moods])
    end_dists = np.array([mood_distance(m, end_mood) for m in track_moods])
    
    # Transition cost matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = mood_distance(track_moods[i], track_moods[j])

    # Dynamic Programming
    dp = np.full((n, n), float('inf'))
    path = np.full((n, n), -1, dtype=int)
    
    # Initialize: paths of length 1 from start
    dp[0, :] = start_dists
    
    # Build DP table
    for k in range(1, n):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cost = dp[k-1, j] + dist_matrix[j, i]
                if cost < dp[k, i]:
                    dp[k, i] = cost
                    path[k, i] = j
    
    # Find best ending track
    total_costs = dp[n-1, :] + end_dists
    last_idx = np.argmin(total_costs)
    min_cost = total_costs[last_idx]
    
    # Backtrack to reconstruct path
    ordered_indices = []
    current = last_idx
    
    for k in range(n - 1, -1, -1):
        ordered_indices.append(int(current))
        if k > 0:
            current = path[k, current]
            if current == -1:
                break
    
    ordered_indices.reverse()
    
    # Calculate flow score (0-1, higher is better)
    max_possible_cost = n * 2.0
    flow_score = max(0, 1 - (min_cost / max_possible_cost))
    
    # Calculate transition smoothness for each step
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
    
    print(f"✅ Flow optimization complete. Score: {flow_score:.3f}")
    
    return {
        "optimizedOrder": ordered_indices,
        "flowScore": float(flow_score),
        "transitions": transitions,
        "totalCost": float(min_cost)
    }


def detect_mood_gaps(tracks: List[Dict], threshold: float = 1.5) -> List[Dict]:
    """
    Detects large mood gaps (jarring transitions) in a playlist.
    
    Args:
        tracks: List of tracks with mood data
        threshold: Distance threshold for detecting gaps
        
    Returns:
        List of gap information with recommendations
    """
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
            # Calculate ideal bridge mood
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
    """
    Calculate overall mood distribution for a playlist.
    """
    mood_counts = {mood: 0 for mood in MOOD_CLASSES}
    
    total = len(tracks)
    if total == 0:
        return {"distribution": {}, "overall_mood": "Unknown", "total_tracks": 0}
    
    for track in tracks:
        # Get mood from either 'mood' or 'moodDetails'
        mood = track.get('mood')
        if isinstance(mood, dict):
            mood = mood.get('fused_mood', 'Calm')
        elif isinstance(mood, str):
            pass  # mood is already a string
        else:
            mood = 'Calm'  # Default
        
        # Map non-existent moods to trained classes
        if mood in MOOD_MAPPING:
            mood = MOOD_MAPPING[mood]
        
        if mood in mood_counts:
            mood_counts[mood] += 1
        else:
            # Default to Calm for unknown moods
            mood_counts['Calm'] += 1
    
    # Convert to percentages
    distribution = {
        mood: round((count / total) * 100, 2)
        for mood, count in mood_counts.items()
        if count > 0
    }
    
    # Determine overall mood (dominant mood)
    if distribution:
        overall_mood = max(distribution, key=distribution.get)
    else:
        overall_mood = "Calm"
    
    return {
        "distribution": distribution,
        "overall_mood": overall_mood,
        "total_tracks": total,
        "mood_diversity": len(distribution),  # Number of different moods
        "dominant_percentage": distribution.get(overall_mood, 0) if distribution else 0
    }