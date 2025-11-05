"""
Live Listening Endpoints
========================
Real-time mood tracking for active listening sessions
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime
from services.live_queue_service import live_queue_service
from services import model_service, music_service

router = APIRouter()


class StartSessionRequest(BaseModel):
    user_id: str


class AddTrackRequest(BaseModel):
    user_id: str
    session_id: str
    track_id: Optional[str] = None
    track_name: str
    artist_name: str


class EndSessionRequest(BaseModel):
    user_id: str
    session_id: str


@router.post("/session/start")
async def start_live_session(request: StartSessionRequest):
    """
    Start a new live listening session
    
    Returns:
        Session ID and initial data
    """
    try:
        session_id = await live_queue_service.start_session(request.user_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "user_id": request.user_id,
            "started_at": datetime.utcnow().isoformat(),
            "message": "Live session started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/add-track")
async def add_track_to_session(request: AddTrackRequest):
    """
    Add track to live session and get updated analytics
    
    Returns:
        Updated queue analytics with real-time mood
    """
    try:
        # Get audio features for track
        features = await music_service.get_audio_features(
            request.track_name,
            request.artist_name
        )
        
        if not features:
            features = music_service.get_default_features()
        
        # Get lyrics sentiment
        from services import lyrics_service
        lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
            request.track_name,
            request.artist_name
        )
        
        # Predict mood for this track
        mood_data = await model_service.predict_mood_from_features(
            features,
            lyrics_sentiment,
            user_id=request.user_id
        )
        
        # Build track data
        track_data = {
            'track_id': request.track_id,
            'name': request.track_name,
            'artist': request.artist_name,
            'features': features,
            'mood': mood_data['primary_mood'],
            'primary_mood': mood_data['primary_mood'],
            'all_moods': mood_data['all_moods'],
            'mood_scores': mood_data['mood_scores'],
            'confidence': mood_data['confidence']
        }
        
        # Add to queue and recalculate
        queue_analytics = await live_queue_service.add_track_to_queue(
            request.user_id,
            request.session_id,
            track_data
        )
        
        return {
            "success": True,
            **queue_analytics,
            "track_added": {
                "name": request.track_name,
                "artist": request.artist_name,
                "mood": mood_data['primary_mood']
            }
        }
        
    except Exception as e:
        print(f"❌ Error adding track to session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{user_id}/current")
async def get_current_session(user_id: str):
    """
    Get current live session analytics
    
    Returns:
        Current queue state and mood analytics
    """
    try:
        # Check for active session
        session_data = await live_queue_service.get_active_session(user_id)
        
        if not session_data:
            return {
                "active": False,
                "message": "No active session"
            }
        
        session_id = session_data['session_id']
        
        # Get full queue data
        queue_data = await live_queue_service.get_current_queue(user_id, session_id)
        
        if not queue_data:
            return {
                "active": False,
                "message": "Session expired"
            }
        
        tracks = queue_data.get('tracks', [])
        
        return {
            "active": True,
            "session_id": session_id,
            "started_at": session_data['started_at'],
            "track_count": len(tracks),
            "aggregated_features": queue_data.get('aggregated_features', {}),
            "current_mood": queue_data.get('current_mood', {}),
            "recent_tracks": [
                {
                    "name": t['name'],
                    "artist": t['artist'],
                    "mood": t['primary_mood'],
                    "played_at": t['played_at']
                }
                for t in tracks[-5:]  # Last 5 tracks
            ],
            "session_duration_minutes": round(
                (datetime.utcnow() - datetime.fromisoformat(
                    session_data['started_at'].replace('Z', '+00:00')
                )).total_seconds() / 60,
                2
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/end")
async def end_live_session(request: EndSessionRequest):
    """
    End live session and save analytics to MongoDB
    
    Returns:
        Final session analytics
    """
    try:
        final_analytics = await live_queue_service.end_session(
            request.user_id,
            request.session_id
        )
        
        if 'error' in final_analytics:
            raise HTTPException(status_code=404, detail=final_analytics['error'])
        
        return {
            "success": True,
            "message": "Session ended and saved to database",
            **final_analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/auto-check/{user_id}")
async def auto_check_session(user_id: str):
    """
    Check if session should be auto-ended due to inactivity
    
    This endpoint should be called periodically by frontend or background task
    
    Returns:
        Status and final analytics if session was ended
    """
    try:
        result = await live_queue_service.check_and_auto_end_session(user_id)
        
        if result:
            return {
                "auto_ended": True,
                "message": "Session ended due to inactivity",
                **result
            }
        
        return {
            "auto_ended": False,
            "message": "Session is still active"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for live listening service"""
    return {
        "status": "healthy",
        "service": "live_listening",
        "features": [
            "Real-time mood tracking",
            "Session management",
            "Auto-save to MongoDB",
            "5-minute inactivity timeout",
            "Live queue analytics"
        ]
    }