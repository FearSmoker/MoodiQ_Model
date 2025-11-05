"""
Train Router - 12 Moods Multi-Tag Compatible
Enhanced user feedback and personalization with extended mood system
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from services import cache_service, model_service, spotify_service

router = APIRouter()


class FeedbackRequest(BaseModel):
    """
    User feedback for mood prediction correction.
    Now supports 12 extended moods + multi-tag feedback
    """
    user_id: str
    track_id: str
    feedback_mood: str  # The correct mood according to user (can be any of 12 moods)
    feedback_moods: Optional[List[str]] = None  # Multiple moods if multi-tag feedback
    playlist_id: Optional[str] = None
    timestamp: Optional[str] = None


class RecommendationRequest(BaseModel):
    """
    Request model for hybrid recommendations with multi-mood support
    """
    seed_tracks: Optional[List[str]] = []
    seed_genres: Optional[List[str]] = []
    target_valence: Optional[float] = None
    target_energy: Optional[float] = None
    target_moods: Optional[List[str]] = None  # NEW: Target specific moods
    user_id: str
    access_token: Optional[str] = None
    limit: int = 20


class NLPCommandRequest(BaseModel):
    """
    Natural language command processing.
    """
    command: str
    context: Optional[Dict[str, Any]] = {}
    user_id: str


class BatchRetrainingRequest(BaseModel):
    """Request for batch retraining with user feedback"""
    user_id: str
    min_samples: int = 10
    force: bool = False


class BehaviorLogRequest(BaseModel):
    """Log implicit user behavior for learning"""
    user_id: str
    track_id: str
    action: str  # "skip", "replay", "like", "add_to_playlist"
    timestamp: Optional[str] = None
    time_of_day: Optional[str] = None  # "morning", "afternoon", "evening", "night"
    current_mood: Optional[str] = None  # User's current mood context


# ============================================
# FEEDBACK SYSTEM (12 MOODS + MULTI-TAG)
# ============================================

@router.post("/feedback")
async def submit_mood_feedback(request: FeedbackRequest):
    """
    Accept user feedback for mood predictions.
    NOW SUPPORTS: 12 extended moods + multi-tag feedback
    
    This endpoint is called when users correct mood predictions.
    """
    try:
        print(f"📝 Received feedback from user {request.user_id}: "
              f"Track {request.track_id} -> {request.feedback_mood}")
        
        # Validate mood using extended mood system
        valid_moods = model_service.ALL_MOOD_LABELS + ["Neutral"]
        if request.feedback_mood not in valid_moods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mood. Must be one of: {', '.join(valid_moods)}"
            )
        
        # Validate multi-mood feedback if provided
        if request.feedback_moods:
            for mood in request.feedback_moods:
                if mood not in valid_moods:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid mood in multi-tag feedback: {mood}"
                    )
        
        # 1. Immediate cache override for this user
        user_override_key = f"user_model:{request.user_id}:track:{request.track_id}"
        
        # Store primary mood
        await cache_service.set_in_cache(
            user_override_key,
            request.feedback_mood,
            expiration=86400 * 30  # Cache for 30 days
        )
        
        # Store multi-mood feedback if provided
        if request.feedback_moods:
            multi_override_key = f"user_model:{request.user_id}:track:{request.track_id}:multi"
            await cache_service.set_in_cache(
                multi_override_key,
                request.feedback_moods,
                expiration=86400 * 30
            )
            print(f"✅ Multi-mood override: {request.feedback_moods}")
        
        print(f"✅ User override cached: {user_override_key} = {request.feedback_mood}")
        
        # 2. Store in feedback log for batch retraining
        feedback_log_key = f"feedback_log:{request.user_id}:{datetime.utcnow().isoformat()}"
        
        feedback_data = {
            "user_id": request.user_id,
            "track_id": request.track_id,
            "feedback_mood": request.feedback_mood,
            "feedback_moods": request.feedback_moods or [request.feedback_mood],
            "playlist_id": request.playlist_id,
            "timestamp": request.timestamp or datetime.utcnow().isoformat()
        }
        
        await cache_service.set_in_cache(
            feedback_log_key,
            feedback_data,
            expiration=86400 * 90  # Keep feedback logs for 90 days
        )
        
        # 3. Update user preference statistics (12 moods)
        user_stats_key = f"user_stats:{request.user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {
            "feedback_count": 0,
            "mood_corrections": {},
            "track_corrections": [],
            "mood_diversity": 0,
            "multi_tag_feedback_count": 0
        }
        
        user_stats["feedback_count"] += 1
        
        # Count primary mood
        if request.feedback_mood not in user_stats["mood_corrections"]:
            user_stats["mood_corrections"][request.feedback_mood] = 0
        user_stats["mood_corrections"][request.feedback_mood] += 1
        
        # Count multi-mood feedback
        if request.feedback_moods:
            user_stats["multi_tag_feedback_count"] += 1
            for mood in request.feedback_moods:
                if mood not in user_stats["mood_corrections"]:
                    user_stats["mood_corrections"][mood] = 0
                user_stats["mood_corrections"][mood] += 1
        
        # Calculate mood diversity (how many different moods user has corrected to)
        user_stats["mood_diversity"] = len(user_stats["mood_corrections"])
        
        # Track individual corrections (keep last 100)
        user_stats["track_corrections"].append({
            "track_id": request.track_id,
            "mood": request.feedback_mood,
            "moods": request.feedback_moods or [request.feedback_mood],
            "timestamp": request.timestamp or datetime.utcnow().isoformat()
        })
        user_stats["track_corrections"] = user_stats["track_corrections"][-100:]
        
        await cache_service.set_in_cache(
            user_stats_key,
            user_stats,
            expiration=86400 * 365  # Keep user stats for 1 year
        )
        
        print(f"📊 User stats updated: {user_stats['feedback_count']} total feedbacks")
        print(f"🎨 Mood diversity: {user_stats['mood_diversity']} different moods")
        
        # 4. Check if user has enough feedback for auto-retraining suggestion
        suggest_retrain = user_stats["feedback_count"] >= 10 and user_stats["feedback_count"] % 10 == 0
        
        return {
            "success": True,
            "message": "Feedback received and applied",
            "track_id": request.track_id,
            "corrected_mood": request.feedback_mood,
            "corrected_moods": request.feedback_moods or [request.feedback_mood],
            "user_feedback_count": user_stats["feedback_count"],
            "mood_diversity": user_stats["mood_diversity"],
            "multi_tag_feedback": request.feedback_moods is not None,
            "suggest_retrain": suggest_retrain,
            "ready_for_personalization": user_stats["feedback_count"] >= 10,
            "mood_system": "12_extended_moods"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing feedback: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "message": "Feedback received but could not be fully processed",
            "error": str(e)
        }


# ============================================
# HYBRID RECOMMENDATIONS (MULTI-MOOD)
# ============================================

@router.post("/recommend")
async def get_hybrid_recommendations(request: RecommendationRequest):
    """
    Generate personalized recommendations using hybrid model.
    NOW SUPPORTS: Multi-mood targeting and 12 extended moods
    """
    try:
        print(f"🎯 Generating recommendations for user {request.user_id}")
        
        # Get user preferences and history
        user_stats_key = f"user_stats:{request.user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {}
        
        # Build personalized recommendations based on user's mood preferences (12 moods)
        user_mood_preferences = user_stats.get("mood_corrections", {})
        
        # Use Spotify's recommendation system as base
        spotify_recommendations = await spotify_service.get_recommendations(
            seed_tracks=request.seed_tracks[:5] if request.seed_tracks else None,
            seed_genres=request.seed_genres[:5] if request.seed_genres else None,
            target_valence=request.target_valence,
            target_energy=request.target_energy,
            limit=request.limit * 2,  # Get more for filtering
            access_token=request.access_token
        )
        
        if not spotify_recommendations:
            print("⚠️ No recommendations from Spotify")
            return {
                "tracks": [],
                "source": "ml_hybrid_multi_mood",
                "user_id": request.user_id,
                "seed_count": len(request.seed_tracks or []),
                "personalized": len(user_mood_preferences) > 0,
                "message": "No recommendations available"
            }
        
        # Enrich recommendations with audio features and mood predictions
        track_ids = [track['id'] for track in spotify_recommendations]
        
        # Get audio features for all recommended tracks
        audio_features_list = await asyncio.gather(
            *[spotify_service.get_audio_features([track_id], request.access_token) 
              for track_id in track_ids[:20]]
        )
        
        # Build enriched recommendations with multi-mood support
        enriched_recommendations = []
        
        for idx, track in enumerate(spotify_recommendations):
            try:
                features = audio_features_list[idx] if idx < len(audio_features_list) else None
                
                if not features:
                    enriched_recommendations.append({
                        **track,
                        "features": None,
                        "primary_mood": "Unknown",
                        "all_moods": []
                    })
                    continue
                
                # Check if user has a preference override for this track
                user_override_key = f"user_model:{request.user_id}:track:{track['id']}"
                cached_mood = await cache_service.get_from_cache(user_override_key)
                
                # Check for multi-mood override
                multi_override_key = f"user_model:{request.user_id}:track:{track['id']}:multi"
                cached_moods = await cache_service.get_from_cache(multi_override_key)
                
                if cached_mood or cached_moods:
                    # Use user preference
                    primary_mood = cached_mood or (cached_moods[0] if cached_moods else "Relaxed")
                    all_moods = cached_moods or [primary_mood]
                    mood_source = "user_preference"
                    confidence = 1.0
                else:
                    # Use multi-mood prediction with similarity matching
                    multi_moods = model_service.get_multi_mood_tags(
                        features,
                        min_similarity=0.65,  # 65% threshold for recommendations
                        max_tags=3
                    )
                    
                    if multi_moods:
                        primary_mood = multi_moods[0][0]
                        all_moods = [mood for mood, _ in multi_moods]
                        confidence = multi_moods[0][1]
                    else:
                        # Fallback to simple rule-based
                        primary_mood = _simple_mood_prediction(features)
                        all_moods = [primary_mood]
                        confidence = 0.5
                    
                    mood_source = "ml_prediction_multi_mood"
                
                # Filter by target moods if specified
                if request.target_moods:
                    # Check if any of the track's moods match target moods
                    if not any(mood in request.target_moods for mood in all_moods):
                        continue  # Skip tracks that don't match target moods
                
                # Calculate relevance score based on user preferences (12 moods)
                relevance_score = 1.0
                if user_mood_preferences:
                    # Boost tracks that match user's preferred moods
                    for mood in all_moods:
                        if mood in user_mood_preferences:
                            mood_preference_count = user_mood_preferences[mood]
                            total_feedback = user_stats.get("feedback_count", 1)
                            preference_ratio = mood_preference_count / total_feedback
                            relevance_score += preference_ratio * 0.3  # Boost per matching mood
                
                enriched_recommendations.append({
                    **track,
                    "features": features,
                    "primary_mood": primary_mood,
                    "all_moods": all_moods,
                    "mood_source": mood_source,
                    "confidence": float(confidence),
                    "relevance_score": float(relevance_score),
                    "personalized": len(user_mood_preferences) > 0,
                    "mood_system": "12_extended_moods"
                })
                
                if len(enriched_recommendations) >= request.limit:
                    break
                
            except Exception as track_error:
                print(f"⚠️ Error enriching track {track.get('id', 'unknown')}: {track_error}")
                continue
        
        # Sort by relevance score if personalized
        if user_mood_preferences:
            enriched_recommendations.sort(key=lambda x: x.get('relevance_score', 1.0), reverse=True)
        
        print(f"✅ Generated {len(enriched_recommendations)} personalized recommendations")
        
        return {
            "tracks": enriched_recommendations,
            "source": "ml_hybrid_multi_mood",
            "user_id": request.user_id,
            "seed_count": len(request.seed_tracks or []),
            "personalized": len(user_mood_preferences) > 0,
            "user_preferences": user_mood_preferences,
            "target_moods": request.target_moods,
            "mood_diversity": len(set([m for t in enriched_recommendations for m in t.get('all_moods', [])])),
            "message": f"Generated {len(enriched_recommendations)} recommendations",
            "mood_system": "12_extended_moods"
        }
        
    except Exception as e:
        print(f"❌ Recommendation generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"ML recommendations unavailable: {str(e)}"
        )


def _simple_mood_prediction(features: Dict) -> str:
    """
    Simple rule-based mood prediction using extended moods.
    Fallback when ML prediction fails.
    """
    valence = features.get('valence', 0.5)
    energy = features.get('energy', 0.5)
    danceability = features.get('danceability', 0.5)
    acousticness = features.get('acousticness', 0.5)
    
    # Map to extended moods based on feature combinations
    if energy > 0.8 and valence > 0.7 and danceability > 0.7:
        return "Party"
    elif valence > 0.75 and energy > 0.6:
        return "Joyful"
    elif valence > 0.65 and energy > 0.7 and danceability > 0.6:
        return "Excited"
    elif energy > 0.7 and valence > 0.5:
        return "Motivated"
    elif valence < 0.3 and energy > 0.7:
        return "Angry"
    elif valence < 0.3 and energy < 0.4:
        return "Melancholic"
    elif valence > 0.5 and energy < 0.4 and acousticness > 0.6:
        return "Romantic"
    elif energy < 0.3 and acousticness > 0.7:
        return "Ambient"
    elif valence < 0.5 and energy < 0.4:
        return "Dreamy"
    elif energy < 0.35:
        return "Relaxed"
    elif valence > 0.55 and energy < 0.5:
        return "Chill"
    elif energy > 0.4 and energy < 0.6:
        return "Focused"
    else:
        return "Relaxed"


# ============================================
# MODEL RETRAINING (12 MOODS)
# ============================================

@router.post("/retrain")
async def trigger_model_retraining(
    request: BatchRetrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger model retraining with accumulated feedback.
    NOW SUPPORTS: 12 extended moods and multi-tag learning
    """
    try:
        user_id = request.user_id
        
        print(f"🔄 Triggering model retraining for user {user_id}")
        
        # Get user stats
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {}
        
        feedback_count = user_stats.get("feedback_count", 0)
        
        if feedback_count < request.min_samples and not request.force:
            return {
                "success": False,
                "message": f"Not enough feedback data for retraining (have {feedback_count}, need {request.min_samples}+)",
                "feedback_count": feedback_count,
                "min_required": request.min_samples
            }
        
        # Add retraining task to background
        background_tasks.add_task(
            _retrain_user_model,
            user_id,
            feedback_count
        )
        
        return {
            "success": True,
            "message": "Retraining started in background",
            "feedback_count": feedback_count,
            "user_id": user_id,
            "estimated_time": "2-5 minutes",
            "status": "processing",
            "mood_system": "12_extended_moods"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Retraining initiation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model retraining failed: {str(e)}"
        )


async def _retrain_user_model(user_id: str, feedback_count: int):
    """
    Background task to retrain user-specific model layer.
    NOW SUPPORTS: 12 extended moods with enhanced feature adjustments
    """
    try:
        print(f"🔬 Starting background retraining for user {user_id}")
        print(f"📚 Collecting {feedback_count} feedback samples...")
        
        # 1. Collect all feedback logs for this user
        feedback_pattern = f"feedback_log:{user_id}:*"
        feedback_keys = await cache_service.get_keys_by_pattern(feedback_pattern, limit=1000)
        
        print(f"📥 Found {len(feedback_keys)} feedback entries")
        
        if len(feedback_keys) < 10:
            print(f"⚠️ Not enough feedback entries found: {len(feedback_keys)}")
            return
        
        # 2. Extract feedback data
        feedback_samples = []
        for key in feedback_keys:
            feedback = await cache_service.get_from_cache(key)
            if feedback:
                feedback_samples.append(feedback)
        
        # 3. Build mood preference weights (12 moods)
        mood_weights = {}
        mood_counts = {}
        
        for sample in feedback_samples:
            # Handle multi-mood feedback
            moods = sample.get('feedback_moods', [sample.get('feedback_mood')])
            for mood in moods:
                if mood:
                    if mood not in mood_counts:
                        mood_counts[mood] = 0
                    mood_counts[mood] += 1
        
        # Normalize to weights (0-1 range)
        total = sum(mood_counts.values())
        for mood, count in mood_counts.items():
            mood_weights[mood] = count / total
        
        print(f"⚖️ Calculated mood weights (12 moods): {mood_weights}")
        
        # 4. Calculate user-specific feature preferences (Enhanced for 12 moods)
        feature_adjustments = {
            'valence_bias': 0.0,
            'energy_bias': 0.0,
            'danceability_bias': 0.0,
            'acousticness_bias': 0.0,
            'instrumentalness_bias': 0.0,
            'tempo_bias': 0.0
        }
        
        # Calculate biases based on mood preferences (12 extended moods)
        for mood, weight in mood_weights.items():
            mood_profile = model_service.EXTENDED_MOODS.get(mood, {}).get('profile', {})
            
            # Apply weighted adjustments based on mood profiles
            if mood == "Joyful":
                feature_adjustments['valence_bias'] += weight * 0.25
                feature_adjustments['energy_bias'] += weight * 0.15
            elif mood == "Excited":
                feature_adjustments['energy_bias'] += weight * 0.3
                feature_adjustments['danceability_bias'] += weight * 0.25
                feature_adjustments['tempo_bias'] += weight * 0.2
            elif mood == "Party":
                feature_adjustments['danceability_bias'] += weight * 0.35
                feature_adjustments['energy_bias'] += weight * 0.25
            elif mood == "Melancholic":
                feature_adjustments['valence_bias'] -= weight * 0.25
                feature_adjustments['energy_bias'] -= weight * 0.2
                feature_adjustments['acousticness_bias'] += weight * 0.15
            elif mood == "Dreamy":
                feature_adjustments['acousticness_bias'] += weight * 0.2
                feature_adjustments['energy_bias'] -= weight * 0.15
                feature_adjustments['instrumentalness_bias'] += weight * 0.15
            elif mood == "Relaxed":
                feature_adjustments['energy_bias'] -= weight * 0.25
                feature_adjustments['acousticness_bias'] += weight * 0.2
            elif mood == "Chill":
                feature_adjustments['energy_bias'] -= weight * 0.15
                feature_adjustments['valence_bias'] += weight * 0.1
            elif mood == "Focused":
                feature_adjustments['energy_bias'] += weight * 0.1
                feature_adjustments['instrumentalness_bias'] += weight * 0.2
            elif mood == "Romantic":
                feature_adjustments['valence_bias'] += weight * 0.15
                feature_adjustments['acousticness_bias'] += weight * 0.2
            elif mood == "Motivated":
                feature_adjustments['energy_bias'] += weight * 0.25
                feature_adjustments['tempo_bias'] += weight * 0.15
            elif mood == "Angry":
                feature_adjustments['energy_bias'] += weight * 0.3
                feature_adjustments['valence_bias'] -= weight * 0.2
            elif mood == "Ambient":
                feature_adjustments['instrumentalness_bias'] += weight * 0.3
                feature_adjustments['energy_bias'] -= weight * 0.25
        
        print(f"🎛️ Feature adjustments: {feature_adjustments}")
        
        # 5. Save personalized model parameters
        personalized_model = {
            "user_id": user_id,
            "trained_at": datetime.utcnow().isoformat(),
            "feedback_count": len(feedback_samples),
            "mood_weights": mood_weights,
            "feature_adjustments": feature_adjustments,
            "model_version": "2.0_multi_mood",
            "mood_system": "12_extended_moods",
            "mood_diversity": len(mood_counts),
            "status": "active"
        }
        
        user_model_key = f"user_model:{user_id}:trained"
        await cache_service.set_in_cache(
            user_model_key,
            personalized_model,
            expiration=86400 * 90  # 90 days
        )
        
        print(f"✅ Personalized model saved for user {user_id}")
        print(f"📊 Model summary:")
        print(f"   - Feedback samples: {len(feedback_samples)}")
        print(f"   - Mood diversity: {len(mood_counts)} different moods")
        print(f"   - Mood preferences: {mood_weights}")
        print(f"   - Feature biases: {feature_adjustments}")
        
        # 6. Update user stats with training completion
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {}
        user_stats["last_trained"] = datetime.utcnow().isoformat()
        user_stats["trained_samples"] = len(feedback_samples)
        user_stats["model_version"] = "2.0_multi_mood"
        await cache_service.set_in_cache(user_stats_key, user_stats, expiration=86400 * 365)
        
        print(f"🎉 Retraining complete for user {user_id}")
        
    except Exception as e:
        print(f"❌ Background retraining failed for user {user_id}: {e}")
        import traceback
        traceback.print_exc()


# ============================================
# USER STATS & ANALYTICS (12 MOODS)
# ============================================

@router.get("/user/{user_id}/stats")
async def get_user_learning_stats(user_id: str):
    """
    Get user's personalization statistics.
    NOW SHOWS: 12 mood distribution and diversity metrics
    """
    try:
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key)
        
        if not user_stats:
            return {
                "user_id": user_id,
                "feedback_count": 0,
                "mood_corrections": {},
                "mood_diversity": 0,
                "personalization_level": "none",
                "has_trained_model": False,
                "track_corrections": [],
                "mood_system": "12_extended_moods"
            }
        
        feedback_count = user_stats.get("feedback_count", 0)
        mood_corrections = user_stats.get("mood_corrections", {})
        mood_diversity = len(mood_corrections)
        
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
        
        # Get recent corrections
        recent_corrections = user_stats.get("track_corrections", [])[-10:]
        
        # Calculate mood distribution percentages
        total_corrections = sum(mood_corrections.values())
        mood_distribution = {
            mood: round((count / total_corrections) * 100, 2) if total_corrections > 0 else 0
            for mood, count in mood_corrections.items()
        }
        
        # Find favorite moods (top 3)
        sorted_moods = sorted(mood_corrections.items(), key=lambda x: x[1], reverse=True)
        favorite_moods = [mood for mood, _ in sorted_moods[:3]]
        
        return {
            "user_id": user_id,
            "feedback_count": feedback_count,
            "mood_corrections": mood_corrections,
            "mood_distribution": mood_distribution,
            "mood_diversity": mood_diversity,
            "favorite_moods": favorite_moods,
            "personalization_level": personalization_level,
            "ready_for_retraining": feedback_count >= 10,
            "has_trained_model": has_trained_model,
            "trained_model_info": trained_model_info if has_trained_model else None,
            "last_trained": user_stats.get("last_trained"),
            "trained_samples": user_stats.get("trained_samples", 0),
            "recent_corrections": recent_corrections,
            "multi_tag_feedback_count": user_stats.get("multi_tag_feedback_count", 0),
            "mood_system": "12_extended_moods",
            "available_moods": model_service.ALL_MOOD_LABELS
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
        
        # Clear trained model
        user_model_key = f"user_model:{user_id}:trained"
        await cache_service.delete_from_cache(user_model_key)
        
        # Clear all user overrides using pattern deletion
        override_pattern = f"user_model:{user_id}:track:*"
        override_deleted = await cache_service.delete_pattern(override_pattern)
        
        # Clear feedback logs
        feedback_pattern = f"feedback_log:{user_id}:*"
        feedback_deleted = await cache_service.delete_pattern(feedback_pattern)
        
        # Clear behavior logs
        behavior_pattern = f"behavior:{user_id}:*"
        behavior_deleted = await cache_service.delete_pattern(behavior_pattern)
        
        print(f"✅ Deleted {override_deleted} track overrides, {feedback_deleted} feedback logs, {behavior_deleted} behavior logs")
        
        return {
            "success": True,
            "message": "User personalization reset successfully",
            "user_id": user_id,
            "deleted_overrides": override_deleted,
            "deleted_feedback_logs": feedback_deleted,
            "deleted_behavior_logs": behavior_deleted,
            "mood_system": "12_extended_moods"
        }
        
    except Exception as e:
        print(f"❌ Error resetting user data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset user data: {str(e)}"
        )

@router.get("/user/{user_id}/model")
async def get_user_personalized_model(user_id: str):
    """
    Get the user's personalized model parameters.
    Shows the learned preferences and adjustments for 12 moods.
    """
    try:
        user_model_key = f"user_model:{user_id}:trained"
        trained_model = await cache_service.get_from_cache(user_model_key)
        
        if not trained_model:
            return {
                "user_id": user_id,
                "has_model": False,
                "message": "No personalized model found. Submit more feedback to train a model.",
                "mood_system": "12_extended_moods",
                "available_moods": model_service.ALL_MOOD_LABELS
            }
        
        # Enhance model info with mood system details
        enhanced_model = {
            **trained_model,
            "mood_system": "12_extended_moods",
            "available_moods": model_service.ALL_MOOD_LABELS,
            "base_moods": model_service.BASE_MOOD_CLASSES,
            "mood_profiles": {
                mood: {
                    "profile": model_service.EXTENDED_MOODS[mood]["profile"],
                    "weight": trained_model.get("mood_weights", {}).get(mood, 0)
                }
                for mood in model_service.ALL_MOOD_LABELS
                if mood in trained_model.get("mood_weights", {})
            }
        }
        
        return {
            "user_id": user_id,
            "has_model": True,
            "model": enhanced_model,
            "message": "Personalized model active with 12 extended moods"
        }
        
    except Exception as e:
        print(f"❌ Error fetching user model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch user model: {str(e)}"
        )

@router.post("/batch-feedback")
async def submit_batch_feedback(requests: List[FeedbackRequest]):
    """
    Submit multiple feedback entries at once.
    Useful for importing historical user data or bulk corrections.
    NOW SUPPORTS: Multi-mood feedback in batch
    """
    try:
        results = []
        
        for request in requests:
            try:
                result = await submit_mood_feedback(request)
                results.append({
                    "track_id": request.track_id,
                    "success": result["success"],
                    "mood": request.feedback_mood,
                    "moods": request.feedback_moods or [request.feedback_mood]
                })
            except Exception as e:
                results.append({
                    "track_id": request.track_id,
                    "success": False,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "total": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "results": results,
            "mood_system": "12_extended_moods"
        }
        
    except Exception as e:
        print(f"❌ Batch feedback failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch feedback processing failed: {str(e)}"
        )

@router.post("/behavior/log")
async def log_user_behavior(request: BehaviorLogRequest):
    """
    Log implicit behavior for learning.
    Tracks user actions to improve recommendations without explicit feedback.
    NOW SUPPORTS: Context-aware logging with mood information
    """
    try:
        behavior_key = f"behavior:{request.user_id}:{datetime.utcnow().date()}"
        
        # Get or create today's behavior log
        behavior_log = await cache_service.get_from_cache(behavior_key) or {
            "skips": [],
            "replays": [],
            "likes": [],
            "add_to_playlist": [],
            "date": str(datetime.utcnow().date())
        }
        
        # Add to appropriate list
        entry = {
            "track_id": request.track_id,
            "timestamp": request.timestamp or datetime.utcnow().isoformat(),
            "time_of_day": request.time_of_day,
            "current_mood": request.current_mood
        }
        
        if request.action == "skip":
            behavior_log["skips"].append(entry)
        elif request.action == "replay":
            behavior_log["replays"].append(entry)
        elif request.action == "like":
            behavior_log["likes"].append(entry)
        elif request.action == "add_to_playlist":
            behavior_log["add_to_playlist"].append(entry)
        
        # Save back
        await cache_service.set_in_cache(behavior_key, behavior_log, expiration=86400*30)
        
        # Analyze patterns for implicit learning
        skip_count = len(behavior_log["skips"])
        replay_count = len(behavior_log["replays"])
        like_count = len(behavior_log["likes"])
        
        # Update user stats with behavior insights
        user_stats_key = f"user_stats:{request.user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key) or {}
        
        if "behavior_patterns" not in user_stats:
            user_stats["behavior_patterns"] = {
                "total_skips": 0,
                "total_replays": 0,
                "total_likes": 0,
                "skip_rate": 0.0
            }
        
        user_stats["behavior_patterns"]["total_skips"] += 1 if request.action == "skip" else 0
        user_stats["behavior_patterns"]["total_replays"] += 1 if request.action == "replay" else 0
        user_stats["behavior_patterns"]["total_likes"] += 1 if request.action == "like" else 0
        
        total_actions = (
            user_stats["behavior_patterns"]["total_skips"] +
            user_stats["behavior_patterns"]["total_replays"] +
            user_stats["behavior_patterns"]["total_likes"]
        )
        
        if total_actions > 0:
            user_stats["behavior_patterns"]["skip_rate"] = (
                user_stats["behavior_patterns"]["total_skips"] / total_actions
            )
        
        await cache_service.set_in_cache(user_stats_key, user_stats, expiration=86400*365)
        
        # Implicit learning: If user skips a lot of a certain mood, adjust preferences
        if skip_count > 15 and request.current_mood:
            # Log pattern for future model adjustment
            pattern_key = f"behavior_pattern:{request.user_id}:skip_mood:{request.current_mood}"
            pattern_count = await cache_service.get_from_cache(pattern_key) or 0
            await cache_service.set_in_cache(
                pattern_key, 
                pattern_count + 1, 
                expiration=86400*30
            )
        
        return {
            "success": True,
            "action_logged": request.action,
            "daily_stats": {
                "skips": skip_count,
                "replays": replay_count,
                "likes": like_count
            },
            "user_behavior_insights": {
                "skip_rate": user_stats["behavior_patterns"]["skip_rate"],
                "total_interactions": total_actions
            },
            "mood_system": "12_extended_moods"
        }
        
    except Exception as e:
        print(f"❌ Error logging behavior: {e}")
        return {
            "success": False,
            "message": "Behavior logged but analysis failed",
            "error": str(e)
        }


@router.get("/user/{user_id}/mood-insights")
async def get_user_mood_insights(user_id: str):
    """
    Get detailed mood insights and patterns for the user.
    Shows which moods they prefer, when they listen to them, etc.
    """
    try:
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key)
        
        if not user_stats:
            return {
                "user_id": user_id,
                "has_insights": False,
                "message": "No insights available yet. Start providing feedback!",
                "mood_system": "12_extended_moods"
            }
        
        mood_corrections = user_stats.get("mood_corrections", {})
        total_corrections = sum(mood_corrections.values())
        
        # Calculate mood preferences as percentages
        mood_preferences = {
            mood: {
                "count": count,
                "percentage": round((count / total_corrections) * 100, 2) if total_corrections > 0 else 0,
                "profile": model_service.EXTENDED_MOODS.get(mood, {}).get("profile", {})
            }
            for mood, count in mood_corrections.items()
        }
        
        # Find dominant mood characteristics
        dominant_moods = sorted(mood_corrections.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Analyze feature preferences across all corrections
        avg_features = {
            "valence": 0.0,
            "energy": 0.0,
            "danceability": 0.0,
            "acousticness": 0.0
        }
        
        for mood, count in mood_corrections.items():
            weight = count / total_corrections if total_corrections > 0 else 0
            mood_profile = model_service.EXTENDED_MOODS.get(mood, {}).get("profile", {})
            
            for feature in avg_features.keys():
                if feature in mood_profile:
                    avg_features[feature] += mood_profile[feature] * weight
        
        # Get behavior patterns
        behavior_patterns = user_stats.get("behavior_patterns", {})
        
        # Check for trained model
        user_model_key = f"user_model:{user_id}:trained"
        trained_model = await cache_service.get_from_cache(user_model_key)
        
        return {
            "user_id": user_id,
            "has_insights": True,
            "total_feedback": total_corrections,
            "mood_diversity": len(mood_corrections),
            "mood_preferences": mood_preferences,
            "dominant_moods": [{"mood": mood, "count": count} for mood, count in dominant_moods],
            "average_feature_preferences": avg_features,
            "behavior_patterns": behavior_patterns,
            "has_personalized_model": trained_model is not None,
            "model_trained_at": trained_model.get("trained_at") if trained_model else None,
            "mood_system": "12_extended_moods",
            "recommendations": _generate_mood_recommendations(mood_corrections, avg_features)
        }
        
    except Exception as e:
        print(f"❌ Error fetching mood insights: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch mood insights: {str(e)}"
        )


def _generate_mood_recommendations(mood_corrections: Dict, avg_features: Dict) -> List[str]:
    """
    Generate personalized mood recommendations based on user patterns.
    """
    recommendations = []
    
    # Sort moods by preference
    sorted_moods = sorted(mood_corrections.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_moods:
        return ["Try different moods to get personalized recommendations!"]
    
    top_mood = sorted_moods[0][0]
    
    # Find similar moods based on feature profiles
    similar_moods = []
    top_mood_profile = model_service.EXTENDED_MOODS.get(top_mood, {}).get("profile", {})
    
    for mood_name in model_service.ALL_MOOD_LABELS:
        if mood_name == top_mood:
            continue
        
        mood_profile = model_service.EXTENDED_MOODS.get(mood_name, {}).get("profile", {})
        
        # Calculate similarity
        similarity = 0
        count = 0
        for feature in ["valence", "energy", "danceability"]:
            if feature in top_mood_profile and feature in mood_profile:
                similarity += 1 - abs(top_mood_profile[feature] - mood_profile[feature])
                count += 1
        
        if count > 0:
            similarity /= count
            if similarity > 0.7:  # 70% similar
                similar_moods.append((mood_name, similarity))
    
    similar_moods.sort(key=lambda x: x[1], reverse=True)
    
    # Generate recommendations
    recommendations.append(f"You love {top_mood} music! Here are some suggestions:")
    
    if similar_moods:
        recommendations.append(f"Try these similar moods: {', '.join([m[0] for m in similar_moods[:3]])}")
    
    # Check for mood diversity
    if len(mood_corrections) < 4:
        recommendations.append("Explore more moods to discover new favorites!")
    
    # Feature-based recommendations
    if avg_features["energy"] > 0.7:
        recommendations.append("You prefer high-energy tracks. Check out 'Excited' and 'Party' moods!")
    elif avg_features["energy"] < 0.3:
        recommendations.append("You enjoy calm music. Try 'Ambient' and 'Dreamy' moods!")
    
    if avg_features["valence"] > 0.7:
        recommendations.append("You're drawn to happy vibes! Explore 'Joyful' mood.")
    elif avg_features["valence"] < 0.3:
        recommendations.append("You appreciate melancholic tones. Try 'Melancholic' mood.")
    
    return recommendations


@router.get("/health")
async def health_check():
    """
    Health check endpoint with extended mood system info
    """
    model_loaded = model_service.mood_model is not None
    cache_connected = cache_service.is_connected()
    
    return {
        "status": "healthy",
        "service": "model_training_feedback_multi_mood",
        "model_loaded": model_loaded,
        "cache_connected": cache_connected,
        "mood_system": "12_extended_moods",
        "base_moods": model_service.BASE_MOOD_CLASSES,
        "extended_moods": model_service.ALL_MOOD_LABELS,
        "total_moods": len(model_service.ALL_MOOD_LABELS),
        "multi_tag_support": True,
        "features": [
            "User Feedback Collection (12 Moods)",
            "Multi-Tag Mood Classification",
            "Personalized Model Training",
            "Hybrid Recommendations",
            "Mood Preference Learning",
            "Batch Processing",
            "Background Training",
            "Behavior Pattern Analysis",
            "Mood Insights & Analytics",
            "Implicit Learning from User Actions"
        ],
        "version": "2.0_multi_mood"
    }


@router.get("/moods/available")
async def get_available_moods():
    """
    Get list of all available moods with their profiles.
    Useful for frontend mood selector UI.
    """
    try:
        moods_info = []
        
        for mood_name in model_service.ALL_MOOD_LABELS:
            mood_config = model_service.EXTENDED_MOODS.get(mood_name, {})
            
            moods_info.append({
                "name": mood_name,
                "base_moods": mood_config.get("base_moods", []),
                "profile": mood_config.get("profile", {}),
                "description": _get_mood_description(mood_name)
            })
        
        return {
            "success": True,
            "mood_system": "12_extended_moods",
            "total_moods": len(moods_info),
            "base_moods": model_service.BASE_MOOD_CLASSES,
            "extended_moods": moods_info
        }
        
    except Exception as e:
        print(f"❌ Error fetching available moods: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch available moods: {str(e)}"
        )


def _get_mood_description(mood_name: str) -> str:
    """
    Get human-readable description for each mood.
    """
    descriptions = {
        "Relaxed": "Calm and peaceful tracks perfect for unwinding",
        "Focused": "Concentration-friendly music for work or study",
        "Romantic": "Soft, emotional songs for intimate moments",
        "Excited": "High-energy tracks that pump you up",
        "Angry": "Intense, aggressive music for releasing emotions",
        "Chill": "Laid-back vibes for casual listening",
        "Melancholic": "Deep, emotional tracks for reflective moods",
        "Dreamy": "Atmospheric, ethereal music for contemplation",
        "Motivated": "Energizing tracks to boost productivity",
        "Joyful": "Happy, uplifting songs that brighten your day",
        "Ambient": "Instrumental soundscapes for background ambiance",
        "Party": "Dance-ready bangers for celebration"
    }
    
    return descriptions.get(mood_name, "Discover this mood's unique vibe")