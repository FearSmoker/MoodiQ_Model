import os
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from . import cache_service, model_service

# MongoDB connection for archiving sessions
import motor.motor_asyncio

class LiveQueueService:
    def __init__(self):
        self.mongo_client = None
        self.db = None
        self.collection = None
        self.memory_cache = {}  # In-memory fallback cache to ensure functionality if Redis is offline/uninitialized

    async def _init_db(self):
        if self.mongo_client:
            return
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            try:
                self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
                self.db = self.mongo_client.get_default_database()
                self.collection = self.db['moodanalytics']
            except Exception as e:
                print(f"⚠️ Failed to connect MongoDB in LiveQueueService: {e}")

    async def start_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        session_key = f"live_session:{user_id}"
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "started_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "active": True
        }
        await cache_service.set_in_cache(session_key, session_data, expiration=3600)
        self.memory_cache[session_key] = session_data
        
        # Initialize empty queue
        queue_key = f"live_queue:{user_id}:{session_id}"
        queue_data = {"tracks": []}
        await cache_service.set_in_cache(queue_key, queue_data, expiration=3600)
        self.memory_cache[queue_key] = queue_data
        
        return session_id

    async def get_active_session(self, user_id: str) -> Optional[Dict]:
        session_key = f"live_session:{user_id}"
        res = await cache_service.get_from_cache(session_key)
        if res:
            return res
        return self.memory_cache.get(session_key)

    async def get_current_queue(self, user_id: str, session_id: str) -> Optional[Dict]:
        queue_key = f"live_queue:{user_id}:{session_id}"
        res = await cache_service.get_from_cache(queue_key)
        if res:
            return res
        return self.memory_cache.get(queue_key)

    async def add_track_to_queue(self, user_id: str, session_id: str, track_data: Dict) -> Dict:
        # Check active session, renew if missing
        session = await self.get_active_session(user_id)
        if not session or session.get("session_id") != session_id:
            session_id = await self.start_session(user_id)
            session = await self.get_active_session(user_id)
            
        # Update last activity
        session["last_activity"] = datetime.utcnow().isoformat()
        session_key = f"live_session:{user_id}"
        await cache_service.set_in_cache(session_key, session, expiration=3600)
        self.memory_cache[session_key] = session
        
        # Get current queue
        queue = await self.get_current_queue(user_id, session_id)
        if not queue:
            queue = {"tracks": []}
            
        # Add played timestamp to track
        track_record = {
            **track_data,
            "played_at": datetime.utcnow().isoformat()
        }
        queue["tracks"].append(track_record)
        
        # Save queue
        queue_key = f"live_queue:{user_id}:{session_id}"
        await cache_service.set_in_cache(queue_key, queue, expiration=3600)
        self.memory_cache[queue_key] = queue
        
        # Calculate queue analytics
        tracks = queue["tracks"]
        track_count = len(tracks)
        
        # Aggregate features and predict queue mood
        features_list = [t.get("features", {}) for t in tracks if t.get("features")]
        if features_list:
            agg_features = {}
            for feat in model_service.MODEL_FEATURES:
                vals = [float(f.get(feat, 0.5)) for f in features_list if f.get(feat) is not None]
                agg_features[feat] = float(np.mean(vals)) if vals else 0.5
                
            pred = model_service.predict_mood_single_track(agg_features)
            current_mood = {
                "primary_mood": pred["primary_mood"],
                "all_moods": pred["all_moods"],
                "mood_scores": pred["mood_scores"],
                "confidence": pred["confidence"]
            }
        else:
            agg_features = {}
            current_mood = {
                "primary_mood": "Relaxed",
                "all_moods": ["Relaxed"],
                "mood_scores": {"Relaxed": 1.0},
                "confidence": 1.0
            }
            
        queue["current_mood"] = current_mood
        queue["aggregated_features"] = agg_features
        await cache_service.set_in_cache(queue_key, queue, expiration=3600)
        self.memory_cache[queue_key] = queue
        
        return {
            "session_id": session_id,
            "track_count": track_count,
            "current_mood": current_mood,
            "aggregated_features": agg_features
        }

    async def end_session(self, user_id: str, session_id: str) -> Dict:
        session_key = f"live_session:{user_id}"
        queue_key = f"live_queue:{user_id}:{session_id}"
        
        session = await self.get_active_session(user_id)
        queue = await self.get_current_queue(user_id, session_id)
        
        if not session or session.get("session_id") != session_id or not queue:
            return {"error": "Session not found or already ended"}
            
        tracks = queue.get("tracks", [])
        track_count = len(tracks)
        
        # Calculate duration
        started_at = datetime.fromisoformat(session["started_at"])
        duration_sec = (datetime.utcnow() - started_at).total_seconds()
        duration_min = round(max(0.1, duration_sec / 60.0), 2)
        
        current_mood = queue.get("current_mood", {
            "primary_mood": "Relaxed",
            "all_moods": ["Relaxed"],
            "mood_scores": {"Relaxed": 1.0},
            "confidence": 1.0
        })
        
        # Clean up cache
        await cache_service.delete_from_cache(session_key)
        await cache_service.delete_from_cache(queue_key)
        self.memory_cache.pop(session_key, None)
        self.memory_cache.pop(queue_key, None)
        
        # Prepare session history record
        record = {
            "session_id": session_id,
            "user_id": user_id,
            "started_at": session["started_at"],
            "ended_at": datetime.utcnow().isoformat(),
            "duration_minutes": duration_min,
            "track_count": track_count,
            "final_mood": current_mood,
            "tracks": tracks,
            "aggregated_features": queue.get("aggregated_features", {})
        }
        
        # Save to MongoDB
        await self._init_db()
        if self.collection is not None:
            try:
                await self.collection.insert_one(record)
                if '_id' in record:
                    del record['_id']
                print(f"✅ Saved live session history to MongoDB")
            except Exception as e:
                print(f"⚠️ Failed to insert session history to MongoDB: {e}")
                
        return {
            "session_id": session_id,
            "user_id": user_id,
            "started_at": session["started_at"],
            "session_duration_minutes": duration_min,
            "track_count": track_count,
            "final_mood": current_mood
        }

    async def check_and_auto_end_session(self, user_id: str) -> Optional[Dict]:
        session = await self.get_active_session(user_id)
        if not session:
            return None
            
        last_activity = datetime.fromisoformat(session["last_activity"])
        inactive_sec = (datetime.utcnow() - last_activity).total_seconds()
        if inactive_sec >= 300:
            print(f"🕒 Session {session['session_id']} inactive for {inactive_sec}s. Auto-ending...")
            return await self.end_session(user_id, session["session_id"])
            
        return None

live_queue_service = LiveQueueService()
