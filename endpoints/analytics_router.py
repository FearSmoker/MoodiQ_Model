"""
Updated Analytics Router - 12-MOOD MULTI-TAG SUPPORT
Compatible with extended mood system and multi-tag classification
"""

from fastapi import APIRouter, HTTPException
from services import cache_service, model_service
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional

router = APIRouter()


@router.get("/user/{user_id}/timeline")
async def get_mood_timeline(user_id: str, days: int = 7):
    """
    Get user's mood trends over time - NOW WITH 12 MOODS
    Uses cached mood predictions with multi-tag support
    """
    try:
        print(f"📊 Generating mood timeline for user {user_id} (last {days} days)")
        
        # Get all user's cached mood predictions
        pattern = f"track:mood:*:{user_id}"
        keys = await cache_service.get_keys_by_pattern(pattern, limit=1000)
        
        if not keys:
            return {
                "user_id": user_id,
                "period_days": days,
                "timeline": [],
                "total_tracked": 0,
                "message": "No mood history found for this user"
            }
        
        mood_history = []
        
        for key in keys:
            try:
                data = await cache_service.get_from_cache(key)
                if not data or not isinstance(data, dict):
                    continue
                
                # Get TTL for timestamp estimation
                ttl = await cache_service.get_ttl(key)
                if ttl > 0:
                    timestamp = datetime.now() - timedelta(seconds=(3600 - ttl))
                else:
                    timestamp = datetime.now()
                
                mood_data = data.get('mood', {})
                
                # Handle both old (single) and new (multi) mood formats
                primary_mood = mood_data.get('primary_mood') or mood_data.get('fused_mood', 'Relaxed')
                all_moods = mood_data.get('all_moods', [primary_mood])
                mood_scores = mood_data.get('mood_scores', {primary_mood: mood_data.get('confidence', 0.5)})
                
                mood_history.append({
                    "primary_mood": primary_mood,
                    "all_moods": all_moods,
                    "mood_scores": mood_scores,
                    "timestamp": timestamp.isoformat(),
                    "track_name": data.get('track_name', 'Unknown'),
                    "artist_name": data.get('artist_name', 'Unknown'),
                    "confidence": mood_data.get('confidence', 0.0)
                })
                
            except Exception as e:
                print(f"⚠️ Error processing cache entry: {e}")
                continue
        
        print(f"✅ Found {len(mood_history)} mood entries")
        
        # Aggregate by day - NOW WITH 12 MOODS
        daily_moods = defaultdict(lambda: {mood: 0 for mood in model_service.ALL_MOOD_LABELS})
        daily_tracks = defaultdict(list)
        
        for entry in mood_history:
            date = entry['timestamp'][:10]  # YYYY-MM-DD
            
            # Count all moods for this track (multi-tag support)
            for mood in entry['all_moods']:
                if mood in daily_moods[date]:
                    daily_moods[date][mood] += 1
            
            daily_tracks[date].append({
                "track": entry['track_name'],
                "artist": entry['artist_name'],
                "mood": entry['primary_mood'],
                "all_moods": entry['all_moods'],
                "confidence": entry['confidence']
            })
        
        # Build timeline
        timeline = []
        for date, moods in sorted(daily_moods.items()):
            total = sum(moods.values())
            
            if total == 0:
                continue
            
            # Calculate percentages
            mood_percentages = {
                mood: round((count / total) * 100, 2)
                for mood, count in moods.items()
                if count > 0
            }
            
            # Get dominant mood
            dominant_mood = max(moods, key=moods.get) if moods else 'Relaxed'
            
            timeline.append({
                "date": date,
                "moods": mood_percentages,
                "total_tracks": len(daily_tracks[date]),
                "total_mood_tags": total,
                "dominant_mood": dominant_mood,
                "mood_diversity": len([m for m, c in moods.items() if c > 0]),
                "tracks": daily_tracks[date]
            })
        
        # Calculate overall statistics
        all_moods_count = defaultdict(int)
        for entry in mood_history:
            for mood in entry['all_moods']:
                all_moods_count[mood] += 1
        
        overall_total = sum(all_moods_count.values())
        overall_distribution = {
            mood: round((count / overall_total) * 100, 2)
            for mood, count in all_moods_count.items()
            if count > 0
        } if overall_total > 0 else {}
        
        return {
            "user_id": user_id,
            "period_days": days,
            "timeline": timeline[-days:],  # Last N days
            "total_tracked": len(mood_history),
            "overall_statistics": {
                "total_tracks_analyzed": len(mood_history),
                "total_mood_tags": overall_total,
                "mood_distribution": overall_distribution,
                "most_common_mood": max(all_moods_count, key=all_moods_count.get) if all_moods_count else 'Unknown',
                "mood_diversity": len(overall_distribution),
                "average_moods_per_track": round(overall_total / len(mood_history), 2) if mood_history else 0
            },
            "mood_system": {
                "total_moods": len(model_service.ALL_MOOD_LABELS),
                "mood_labels": model_service.ALL_MOOD_LABELS,
                "multi_tag_enabled": True
            }
        }
        
    except Exception as e:
        print(f"❌ Error generating mood timeline: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/mood-distribution")
async def get_mood_distribution(user_id: str):
    """
    Get overall mood distribution for user - 12 MOODS
    Shows which moods user listens to most
    """
    try:
        print(f"📊 Analyzing mood distribution for user {user_id}")
        
        pattern = f"track:mood:*:{user_id}"
        keys = await cache_service.get_keys_by_pattern(pattern, limit=1000)
        
        if not keys:
            return {
                "user_id": user_id,
                "distribution": {},
                "total_tracks": 0,
                "message": "No mood data found"
            }
        
        mood_counts = defaultdict(int)
        track_count = 0
        mood_scores_sum = defaultdict(float)
        
        for key in keys:
            try:
                data = await cache_service.get_from_cache(key)
                if not data or not isinstance(data, dict):
                    continue
                
                mood_data = data.get('mood', {})
                
                # Count all moods (multi-tag)
                all_moods = mood_data.get('all_moods', [])
                mood_scores = mood_data.get('mood_scores', {})
                
                if not all_moods:
                    primary_mood = mood_data.get('primary_mood') or mood_data.get('fused_mood', 'Relaxed')
                    all_moods = [primary_mood]
                    mood_scores = {primary_mood: mood_data.get('confidence', 0.5)}
                
                for mood in all_moods:
                    mood_counts[mood] += 1
                    mood_scores_sum[mood] += mood_scores.get(mood, 0.5)
                
                track_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing entry: {e}")
                continue
        
        if track_count == 0:
            return {
                "user_id": user_id,
                "distribution": {},
                "total_tracks": 0
            }
        
        total_mood_tags = sum(mood_counts.values())
        
        # Calculate percentages and average confidence
        distribution = {}
        for mood, count in mood_counts.items():
            distribution[mood] = {
                "count": count,
                "percentage": round((count / total_mood_tags) * 100, 2),
                "avg_confidence": round(mood_scores_sum[mood] / count, 2) if count > 0 else 0
            }
        
        # Sort by count
        sorted_distribution = dict(sorted(
            distribution.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ))
        
        return {
            "user_id": user_id,
            "total_tracks": track_count,
            "total_mood_tags": total_mood_tags,
            "avg_moods_per_track": round(total_mood_tags / track_count, 2),
            "distribution": sorted_distribution,
            "top_3_moods": list(sorted_distribution.keys())[:3],
            "mood_diversity": len(distribution),
            "mood_system": {
                "total_available_moods": len(model_service.ALL_MOOD_LABELS),
                "moods_used": len(distribution)
            }
        }
        
    except Exception as e:
        print(f"❌ Error getting mood distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/mood-patterns")
async def get_mood_patterns(user_id: str):
    """
    Analyze mood listening patterns - 12 MOODS
    Shows which moods tend to appear together (multi-tag co-occurrence)
    """
    try:
        print(f"🔍 Analyzing mood patterns for user {user_id}")
        
        pattern = f"track:mood:*:{user_id}"
        keys = await cache_service.get_keys_by_pattern(pattern, limit=1000)
        
        if not keys:
            return {
                "user_id": user_id,
                "patterns": {},
                "message": "No data found"
            }
        
        # Track mood co-occurrences
        mood_pairs = defaultdict(int)
        single_moods = defaultdict(int)
        
        for key in keys:
            try:
                data = await cache_service.get_from_cache(key)
                if not data or not isinstance(data, dict):
                    continue
                
                mood_data = data.get('mood', {})
                all_moods = mood_data.get('all_moods', [])
                
                if not all_moods:
                    continue
                
                # Track single mood occurrences
                for mood in all_moods:
                    single_moods[mood] += 1
                
                # Track mood pairs (co-occurrence)
                if len(all_moods) > 1:
                    for i, mood1 in enumerate(all_moods):
                        for mood2 in all_moods[i+1:]:
                            pair = tuple(sorted([mood1, mood2]))
                            mood_pairs[pair] += 1
                
            except Exception as e:
                continue
        
        # Calculate co-occurrence percentages
        patterns = []
        for (mood1, mood2), count in sorted(mood_pairs.items(), key=lambda x: x[1], reverse=True):
            # Calculate how often these moods appear together vs individually
            mood1_total = single_moods[mood1]
            mood2_total = single_moods[mood2]
            
            co_occurrence_rate = round((count / min(mood1_total, mood2_total)) * 100, 2)
            
            patterns.append({
                "moods": [mood1, mood2],
                "co_occurrence_count": count,
                "co_occurrence_rate": co_occurrence_rate,
                "mood1_total": mood1_total,
                "mood2_total": mood2_total
            })
        
        return {
            "user_id": user_id,
            "total_tracks_analyzed": len(keys),
            "patterns": patterns[:10],  # Top 10 patterns
            "single_mood_counts": dict(sorted(
                single_moods.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            "insights": {
                "most_common_pair": patterns[0]["moods"] if patterns else None,
                "total_unique_pairs": len(patterns)
            }
        }
        
    except Exception as e:
        print(f"❌ Error analyzing mood patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/feedback-stats")
async def get_feedback_stats(user_id: str):
    """
    Get user feedback statistics - UPDATED FOR 12 MOODS
    Shows how user corrections impact the model
    """
    try:
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key)
        
        if not user_stats:
            return {
                "user_id": user_id,
                "feedback_count": 0,
                "mood_corrections": {},
                "message": "No feedback data found"
            }
        
        mood_corrections = user_stats.get('mood_corrections', {})
        total_feedback = user_stats.get('feedback_count', 0)
        
        # Calculate correction rates for each mood
        correction_analysis = {}
        for mood, count in mood_corrections.items():
            correction_analysis[mood] = {
                "correction_count": count,
                "percentage": round((count / total_feedback) * 100, 2) if total_feedback > 0 else 0
            }
        
        # Sort by correction count
        sorted_corrections = dict(sorted(
            correction_analysis.items(),
            key=lambda x: x[1]['correction_count'],
            reverse=True
        ))
        
        return {
            "user_id": user_id,
            "feedback_count": total_feedback,
            "mood_corrections": sorted_corrections,
            "most_corrected_to": max(mood_corrections, key=mood_corrections.get) if mood_corrections else None,
            "personalization_ready": total_feedback >= 5,
            "mood_preferences": {
                "favorite_moods": list(sorted_corrections.keys())[:3],
                "mood_diversity": len(mood_corrections)
            }
        }
        
    except Exception as e:
        print(f"❌ Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/global/mood-trends")
async def get_global_mood_trends(limit: int = 100):
    """
    Get global mood trends across all users - 12 MOODS
    Shows most popular moods overall
    """
    try:
        print(f"🌍 Analyzing global mood trends")
        
        # Get all mood cache entries
        pattern = "track:mood:*"
        keys = await cache_service.get_keys_by_pattern(pattern, limit=limit)
        
        if not keys:
            return {
                "mood_distribution": {},
                "total_tracks": 0,
                "message": "No global data found"
            }
        
        mood_counts = defaultdict(int)
        track_count = 0
        
        for key in keys:
            try:
                data = await cache_service.get_from_cache(key)
                if not data or not isinstance(data, dict):
                    continue
                
                mood_data = data.get('mood', {})
                all_moods = mood_data.get('all_moods', [])
                
                if not all_moods:
                    primary_mood = mood_data.get('primary_mood') or mood_data.get('fused_mood', 'Relaxed')
                    all_moods = [primary_mood]
                
                for mood in all_moods:
                    mood_counts[mood] += 1
                
                track_count += 1
                
            except Exception as e:
                continue
        
        total_mood_tags = sum(mood_counts.values())
        
        # Calculate percentages
        distribution = {
            mood: {
                "count": count,
                "percentage": round((count / total_mood_tags) * 100, 2)
            }
            for mood, count in mood_counts.items()
        }
        
        sorted_distribution = dict(sorted(
            distribution.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ))
        
        return {
            "total_tracks_analyzed": track_count,
            "total_mood_tags": total_mood_tags,
            "avg_moods_per_track": round(total_mood_tags / track_count, 2) if track_count > 0 else 0,
            "mood_distribution": sorted_distribution,
            "top_5_moods": list(sorted_distribution.keys())[:5],
            "mood_diversity": len(distribution),
            "mood_system": {
                "total_moods": len(model_service.ALL_MOOD_LABELS),
                "moods_in_use": len(distribution)
            }
        }
        
    except Exception as e:
        print(f"❌ Error getting global trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for analytics service"""
    return {
        "status": "healthy",
        "service": "analytics",
        "features": [
            "User mood timeline (12 moods)",
            "Mood distribution analysis",
            "Mood pattern detection",
            "Multi-tag co-occurrence analysis",
            "Feedback statistics",
            "Global mood trends"
        ],
        "mood_system": {
            "total_moods": len(model_service.ALL_MOOD_LABELS),
            "mood_labels": model_service.ALL_MOOD_LABELS,
            "base_moods": model_service.BASE_MOOD_CLASSES,
            "multi_tag_support": True
        }
    }