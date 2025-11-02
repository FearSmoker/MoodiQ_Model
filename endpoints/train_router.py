from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from services import cache_service, model_service, spotify_service

router = APIRouter()


class FeedbackRequest(BaseModel):
    """
    User feedback for mood prediction correction.
    Matches backend's POST /api/user/feedback
    """
    user_id: str
    track_id: str
    feedback_mood: str  # The correct mood according to user
    playlist_id: Optional[str] = None
    timestamp: Optional[str] = None


class RecommendationRequest(BaseModel):
    """
    Request model matching backend's POST /api/playlists/recommendations
    """
    seed_tracks: Optional[List[str]] = []
    seed_genres: Optional[List[str]] = []
    target_valence: Optional[float] = None
    target_energy: Optional[float] = None
    user_id: str
    access_token: Optional[str] = None
    limit: int = 20


class NLPCommandRequest(BaseModel):
    """
    Natural language command processing.
    Matches backend's POST /api/user/voice-command
    """
    command: str
    context: Optional[Dict[str, Any]] = {}
    user_id: str


@router.post("/feedback")
async def submit_mood_feedback(request: FeedbackRequest):
    """
    Accept user feedback for mood predictions.
    Implements incremental learning by caching user corrections.
    
    This endpoint is called by the backend when users correct mood predictions.
    """
    try:
        print(f"📝 Received feedback from user {request.user_id}: "
              f"Track {request.track_id} -> {request.feedback_mood}")
        
        # Validate mood using the MOOD_CLASSES from model_service
        valid_moods = model_service.MOOD_CLASSES + ["Neutral"]
        if request.feedback_mood not in valid_moods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mood. Must be one of: {', '.join(valid_moods)}"
            )
        
        # 1. Immediate cache override for this user
        # This ensures the next time they request this track, they get their correction
        user_override_key = f"user_model:{request.user_id}:track:{request.track_id}"
        
        await cache_service.set_in_cache(
            user_override_key,
            request.feedback_mood,
            expiration=86400 * 30  # Cache for 30 days
        )
        
        print(f"✅ User override cached: {user_override_key} = {request.feedback_mood}")
        
        # 2. Store in feedback log for batch retraining
        # In production, this would go to a database (MongoDB, PostgreSQL, etc.)
        feedback_log_key = f"feedback_log:{request.user_id}:{datetime.utcnow().isoformat()}"
        
        feedback_data = {
            "user_id": request.user_id,
            "track_id": request.track_id,
            "feedback_mood": request.feedback_mood,
            "playlist_id": request.playlist_id,
            "timestamp": request.timestamp or datetime.utcnow().isoformat()
        }
        
        await cache_service.set_in_cache(
            feedback_log_key,
            feedback_data,
            expiration=86400 * 90  # Keep feedback logs for 90 days
        )
        
        # 3. Update user preference statistics
        user_stats_key = f"user_stats:{request.user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {
            "feedback_count": 0,
            "mood_corrections": {}
        }
        
        user_stats["feedback_count"] += 1
        
        if request.feedback_mood not in user_stats["mood_corrections"]:
            user_stats["mood_corrections"][request.feedback_mood] = 0
        user_stats["mood_corrections"][request.feedback_mood] += 1
        
        await cache_service.set_in_cache(
            user_stats_key,
            user_stats,
            expiration=86400 * 365  # Keep user stats for 1 year
        )
        
        print(f"📊 User stats updated: {user_stats['feedback_count']} total feedbacks")
        
        # 4. TODO: In production, publish to message queue (RabbitMQ, Redis Pub/Sub)
        # for a separate training worker to process batch updates
        # Example: await queue_service.publish("feedback_queue", feedback_data)
        
        return {
            "success": True,
            "message": "Feedback received and applied",
            "track_id": request.track_id,
            "corrected_mood": request.feedback_mood,
            "user_feedback_count": user_stats["feedback_count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing feedback: {e}")
        import traceback
        traceback.print_exc()
        
        # Don't fail the request - feedback is important but not critical
        return {
            "success": False,
            "message": "Feedback received but could not be fully processed",
            "error": str(e)
        }


@router.post("/recommend")
async def get_hybrid_recommendations(request: RecommendationRequest):
    """
    Generate personalized recommendations using hybrid model.
    Combines content-based filtering (audio features) with collaborative filtering (user history).
    
    Called by backend when it needs ML-based recommendations.
    """
    try:
        print(f"🎯 Generating recommendations for user {request.user_id}")
        
        # Get user preferences and history
        user_stats_key = f"user_stats:{request.user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {}
        
        # Build personalized recommendations based on user's mood preferences
        user_mood_preferences = user_stats.get("mood_corrections", {})
        
        # Use Spotify's recommendation system as base
        spotify_recommendations = await spotify_service.get_recommendations(
            seed_tracks=request.seed_tracks[:5] if request.seed_tracks else None,
            seed_genres=request.seed_genres[:5] if request.seed_genres else None,
            target_valence=request.target_valence,
            target_energy=request.target_energy,
            limit=request.limit,
            access_token=request.access_token
        )
        
        if not spotify_recommendations:
            print("⚠️ No recommendations from Spotify")
            return {
                "tracks": [],
                "source": "ml_hybrid",
                "user_id": request.user_id,
                "seed_count": len(request.seed_tracks or []),
                "personalized": len(user_mood_preferences) > 0,
                "message": "No recommendations available"
            }
        
        # Enrich recommendations with audio features and mood predictions
        track_ids = [track['id'] for track in spotify_recommendations]
        
        # Get audio features for all recommended tracks
        audio_features_list = await spotify_service.get_audio_features(
            track_ids,
            request.access_token
        )
        
        # Build enriched recommendations
        enriched_recommendations = []
        
        for idx, track in enumerate(spotify_recommendations):
            try:
                features = audio_features_list[idx] if idx < len(audio_features_list) else None
                
                if not features:
                    # Include track without features
                    enriched_recommendations.append({
                        **track,
                        "features": None,
                        "predicted_mood": "Unknown"
                    })
                    continue
                
                # Check if user has a preference override for this track
                user_override_key = f"user_model:{request.user_id}:track:{track['id']}"
                cached_mood = await cache_service.get_from_cache(user_override_key)
                
                if cached_mood:
                    predicted_mood = cached_mood
                    mood_source = "user_preference"
                else:
                    # Use model to predict mood (without lyrics for faster recommendations)
                    # Simple classification based on features
                    valence = features.get('valence', 0.5)
                    energy = features.get('energy', 0.5)
                    
                    if valence > 0.6 and energy > 0.6:
                        predicted_mood = "Happy"
                    elif valence > 0.6 and energy <= 0.4:
                        predicted_mood = "Calm"
                    elif valence <= 0.4 and energy > 0.6:
                        predicted_mood = "Energetic"
                    elif valence <= 0.4 and energy <= 0.4:
                        predicted_mood = "Sad"
                    else:
                        predicted_mood = "Neutral"
                    
                    mood_source = "ml_prediction"
                
                # Calculate relevance score based on user preferences
                relevance_score = 1.0
                if user_mood_preferences and predicted_mood in user_mood_preferences:
                    # Boost tracks that match user's preferred moods
                    mood_preference_count = user_mood_preferences[predicted_mood]
                    total_feedback = user_stats.get("feedback_count", 1)
                    preference_ratio = mood_preference_count / total_feedback
                    relevance_score = 1.0 + (preference_ratio * 0.5)  # Up to 50% boost
                
                enriched_recommendations.append({
                    **track,
                    "features": features,
                    "predicted_mood": predicted_mood,
                    "mood_source": mood_source,
                    "relevance_score": float(relevance_score),
                    "personalized": len(user_mood_preferences) > 0
                })
                
            except Exception as track_error:
                print(f"⚠️ Error enriching track {track.get('id', 'unknown')}: {track_error}")
                # Include track without enrichment
                enriched_recommendations.append({
                    **track,
                    "features": None,
                    "predicted_mood": "Unknown"
                })
                continue
        
        # Sort by relevance score if personalized
        if user_mood_preferences:
            enriched_recommendations.sort(key=lambda x: x.get('relevance_score', 1.0), reverse=True)
        
        print(f"✅ Generated {len(enriched_recommendations)} personalized recommendations")
        
        return {
            "tracks": enriched_recommendations,
            "source": "ml_hybrid",
            "user_id": request.user_id,
            "seed_count": len(request.seed_tracks or []),
            "personalized": len(user_mood_preferences) > 0,
            "user_preferences": user_mood_preferences,
            "message": f"Generated {len(enriched_recommendations)} recommendations"
        }
        
    except Exception as e:
        print(f"❌ Recommendation generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Let backend fall back to Spotify recommendations
        raise HTTPException(
            status_code=500,
            detail=f"ML recommendations unavailable: {str(e)}"
        )


@router.post("/retrain")
async def trigger_model_retraining(request: Dict[str, Any]):
    """
    Trigger model retraining with accumulated feedback.
    This would typically be called by an admin or scheduled job.
    """
    try:
        user_id = request.get('user_id')
        
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required"
            )
        
        print(f"🔄 Triggering model retraining for user {user_id}")
        
        # Get user stats
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {}
        
        feedback_count = user_stats.get("feedback_count", 0)
        
        if feedback_count < 10:
            return {
                "success": False,
                "message": f"Not enough feedback data for retraining (have {feedback_count}, need 10+)",
                "feedback_count": feedback_count
            }
        
        # TODO: Implement actual retraining logic
        # This would:
        # 1. Fetch all feedback logs for the user
        # 2. Extract audio features for those tracks using spotify_service
        # 3. Fine-tune a user-specific model layer using model_service
        # 4. Save the updated model weights
        
        # For now, simulate retraining by updating cache
        print(f"📚 Collecting feedback data for {feedback_count} corrections...")
        print(f"🔬 Training user-specific model layer...")
        print(f"💾 Saving model checkpoint...")
        
        # Mark user as having a trained model
        user_model_key = f"user_model:{user_id}:trained"
        await cache_service.set_in_cache(
            user_model_key,
            {
                "trained_at": datetime.utcnow().isoformat(),
                "feedback_count": feedback_count,
                "version": "1.0"
            },
            expiration=86400 * 30  # 30 days
        )
        
        return {
            "success": True,
            "message": "Retraining completed successfully (placeholder - implement actual training)",
            "feedback_count": feedback_count,
            "user_id": user_id,
            "model_version": "1.0"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model retraining failed: {str(e)}"
        )


@router.get("/user/{user_id}/stats")
async def get_user_learning_stats(user_id: str):
    """
    Get user's personalization statistics.
    Shows how much the model has learned from their feedback.
    """
    try:
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key)
        
        if not user_stats:
            return {
                "user_id": user_id,
                "feedback_count": 0,
                "mood_corrections": {},
                "personalization_level": "none",
                "has_trained_model": False
            }
        
        feedback_count = user_stats.get("feedback_count", 0)
        
        # Determine personalization level
        if feedback_count >= 50:
            personalization_level = "high"
        elif feedback_count >= 20:
            personalization_level = "medium"
        elif feedback_count >= 5:
            personalization_level = "low"
        else:
            personalization_level = "minimal"
        
        # Check if user has a trained model
        user_model_key = f"user_model:{user_id}:trained"
        trained_model_info = await cache_service.get_from_cache(user_model_key)
        has_trained_model = trained_model_info is not None
        
        return {
            "user_id": user_id,
            "feedback_count": feedback_count,
            "mood_corrections": user_stats.get("mood_corrections", {}),
            "personalization_level": personalization_level,
            "ready_for_retraining": feedback_count >= 10,
            "has_trained_model": has_trained_model,
            "trained_model_info": trained_model_info if has_trained_model else None
        }
        
    except Exception as e:
        print(f"❌ Error fetching user stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user stats: {str(e)}"
        )


@router.delete("/user/{user_id}/reset")
async def reset_user_personalization(user_id: str):
    """
    Reset user's personalization data.
    Useful for testing or if user wants to start fresh.
    """
    try:
        print(f"🗑️ Resetting personalization for user {user_id}")
        
        # Clear user stats
        user_stats_key = f"user_stats:{user_id}"
        await cache_service.delete_from_cache(user_stats_key)
        
        # Clear trained model flag
        user_model_key = f"user_model:{user_id}:trained"
        await cache_service.delete_from_cache(user_model_key)
        
        # Clear all user overrides using pattern deletion
        pattern = f"user_model:{user_id}:track:*"
        deleted_count = await cache_service.delete_pattern(pattern)
        
        # Clear feedback logs
        feedback_pattern = f"feedback_log:{user_id}:*"
        feedback_deleted = await cache_service.delete_pattern(feedback_pattern)
        
        print(f"✅ Deleted {deleted_count} track overrides and {feedback_deleted} feedback logs")
        
        return {
            "success": True,
            "message": "User personalization reset successfully",
            "user_id": user_id,
            "deleted_overrides": deleted_count,
            "deleted_feedback_logs": feedback_deleted
        }
        
    except Exception as e:
        print(f"❌ Error resetting user data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset user data: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model_service.mood_model is not None
    cache_connected = cache_service.is_connected()
    
    return {
        "status": "healthy",
        "service": "model_training_feedback",
        "model_loaded": model_loaded,
        "cache_connected": cache_connected
    }