from fastapi import APIRouter
from services import cache_service
from datetime import datetime, timedelta
from collections import defaultdict

router = APIRouter()

@router.get("/user/{user_id}/timeline")
async def get_mood_timeline(user_id: str, days: int = 7):
    """
    Get user's mood trends over time
    Uses cached mood predictions
    """
    # Get all user's cached mood predictions
    pattern = f"track:mood:*:{user_id}"
    keys = await cache_service.get_keys_by_pattern(pattern, limit=1000)
    
    mood_history = []
    for key in keys:
        data = await cache_service.get_from_cache(key)
        if data and isinstance(data, dict):
            # Try to get timestamp from cache TTL
            ttl = await cache_service.get_ttl(key)
            if ttl > 0:
                timestamp = datetime.now() - timedelta(seconds=(3600 - ttl))
                mood_history.append({
                    "mood": data.get('mood', {}).get('fused_mood', 'Unknown'),
                    "timestamp": timestamp.isoformat(),
                    "track_id": key.split(':')[2]
                })
    
    # Aggregate by day
    daily_moods = defaultdict(lambda: {"Happy": 0, "Sad": 0, "Calm": 0, "Energetic": 0})
    
    for entry in mood_history:
        date = entry['timestamp'][:10]  # YYYY-MM-DD
        mood = entry['mood']
        if mood in daily_moods[date]:
            daily_moods[date][mood] += 1
    
    timeline = []
    for date, moods in sorted(daily_moods.items()):
        total = sum(moods.values())
        timeline.append({
            "date": date,
            "moods": {m: (count/total)*100 for m, count in moods.items()},
            "total_tracks": total,
            "dominant_mood": max(moods, key=moods.get)
        })
    
    return {
        "user_id": user_id,
        "period_days": days,
        "timeline": timeline[-days:],  # Last N days
        "total_tracked": len(mood_history)
    }