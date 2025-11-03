from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services import spotify_service, model_service

router = APIRouter()

class GeneratePlaylistRequest(BaseModel):
    target_mood: str  # "Happy", "Sad", "Calm", "Energetic"
    user_id: str
    access_token: str
    limit: int = 20
    seed_tracks: Optional[List[str]] = []

@router.post("/playlist")
async def generate_mood_playlist(request: GeneratePlaylistRequest):
    """
    Generate playlist for target mood using existing model
    """
    # 1. Map mood to audio features
    mood_profiles = {
        "Happy": {"target_valence": 0.8, "target_energy": 0.7},
        "Sad": {"target_valence": 0.2, "target_energy": 0.3},
        "Calm": {"target_valence": 0.5, "target_energy": 0.3},
        "Energetic": {"target_valence": 0.7, "target_energy": 0.9}
    }
    
    if request.target_mood not in mood_profiles:
        raise HTTPException(400, f"Invalid mood. Choose from: {list(mood_profiles.keys())}")
    
    profile = mood_profiles[request.target_mood]
    
    # 2. Get recommendations from Spotify
    recommendations = await spotify_service.get_recommendations(
        seed_tracks=request.seed_tracks[:5] if request.seed_tracks else None,
        target_valence=profile["target_valence"],
        target_energy=profile["target_energy"],
        limit=request.limit * 2,  # Get more, filter later
        access_token=request.access_token
    )
    
    # 3. Get audio features
    track_ids = [track['id'] for track in recommendations]
    audio_features = await spotify_service.get_audio_features(track_ids, request.access_token)
    
    # 4. Use YOUR MODEL to verify mood matches
    filtered_tracks = []
    for i, track in enumerate(recommendations):
        features = audio_features[i]
        if not features:
            continue
        
        # Predict mood using your trained model
        mood_data = await model_service.predict_mood_from_features(
            features, 
            {"polarity": 0.0, "subjectivity": 0.0},  # No lyrics for speed
            user_id=request.user_id
        )
        
        # Keep only tracks that match target mood
        if mood_data['fused_mood'] == request.target_mood:
            track['predicted_mood'] = mood_data['fused_mood']
            track['confidence'] = mood_data['confidence']
            track['features'] = features
            filtered_tracks.append(track)
        
        if len(filtered_tracks) >= request.limit:
            break
    
    # 5. Optimize flow
    if len(filtered_tracks) > 1:
        optimization = model_service.optimize_flow_dp(
            filtered_tracks,
            profile,  # Start mood
            profile   # End mood
        )
        
        # Reorder tracks
        ordered_tracks = [filtered_tracks[i] for i in optimization['optimizedOrder']]
    else:
        ordered_tracks = filtered_tracks
    
    return {
        "target_mood": request.target_mood,
        "tracks": ordered_tracks,
        "total": len(ordered_tracks),
        "flow_score": optimization.get('flowScore', 1.0) if len(filtered_tracks) > 1 else 1.0
    }

class GenerateActivityRequest(BaseModel):
    activity: str  # "study", "workout", "party", "sleep", "work", "meditation"
    user_id: str
    access_token: str
    limit: int = 20

@router.post("/activity")
async def generate_activity_playlist(request: GenerateActivityRequest):
    """
    Generate playlist for specific activity
    """
    # Activity → Mood + Audio Feature Mapping
    activity_profiles = {
        "study": {
            "mood": "Calm",
            "target_valence": 0.4,
            "target_energy": 0.3,
            "target_instrumentalness": 0.7,  # Prefer instrumental
            "target_acousticness": 0.6
        },
        "workout": {
            "mood": "Energetic",
            "target_valence": 0.7,
            "target_energy": 0.95,
            "target_tempo": 140,
            "target_danceability": 0.8
        },
        "party": {
            "mood": "Happy",
            "target_valence": 0.9,
            "target_energy": 0.85,
            "target_danceability": 0.9
        },
        "sleep": {
            "mood": "Calm",
            "target_valence": 0.3,
            "target_energy": 0.15,
            "target_acousticness": 0.8,
            "target_tempo": 60
        },
        "meditation": {
            "mood": "Calm",
            "target_valence": 0.5,
            "target_energy": 0.2,
            "target_instrumentalness": 0.9
        },
        "work": {
            "mood": "Calm",
            "target_valence": 0.6,
            "target_energy": 0.5,
            "target_instrumentalness": 0.5
        }
    }
    
    activity_lower = request.activity.lower()
    if activity_lower not in activity_profiles:
        raise HTTPException(400, f"Unknown activity. Choose from: {list(activity_profiles.keys())}")
    
    profile = activity_profiles[activity_lower]
    target_mood = profile.pop("mood")
    
    # Get recommendations with activity-specific features
    recommendations = await spotify_service.get_recommendations(
        limit=request.limit * 2,
        access_token=request.access_token,
        **profile
    )
    
    # Filter using model (same as generate_mood_playlist)
    track_ids = [track['id'] for track in recommendations]
    audio_features = await spotify_service.get_audio_features(track_ids, request.access_token)
    
    filtered_tracks = []
    for i, track in enumerate(recommendations):
        features = audio_features[i]
        if not features:
            continue
        
        mood_data = await model_service.predict_mood_from_features(
            features, 
            {"polarity": 0.0, "subjectivity": 0.0},
            user_id=request.user_id
        )
        
        track['predicted_mood'] = mood_data['fused_mood']
        track['confidence'] = mood_data['confidence']
        track['features'] = features
        track['activity_match'] = mood_data['fused_mood'] == target_mood
        
        filtered_tracks.append(track)
        
        if len(filtered_tracks) >= request.limit:
            break
    
    return {
        "activity": request.activity,
        "target_mood": target_mood,
        "tracks": filtered_tracks,
        "total": len(filtered_tracks)
    }