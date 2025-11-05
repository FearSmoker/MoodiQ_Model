"""
Live Listening Queue Service
=============================
Real-time mood tracking using Redis for live playback sessions
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from . import cache_service


class LiveQueueService:
    """
    Manages live listening queues with real-time mood analytics
    """
    
    # Session configuration
    SESSION_TIMEOUT = 300  # 5 minutes in seconds
    MAX_QUEUE_SIZE = 100   # Maximum tracks in queue
    
    @staticmethod
    def _get_queue_key(user_id: str, session_id: str) -> str:
        """Generate Redis key for live queue"""
        return f"live_queue:{user_id}:{session_id}"
    
    @staticmethod
    def _get_session_key(user_id: str) -> str:
        """Generate Redis key for active session"""
        return f"live_session:{user_id}:active"
    
    async def start_session(self, user_id: str) -> str:
        """
        Start a new live listening session
        
        Args:
            user_id: User identifier
            
        Returns:
            Session ID
        """
        # Generate session ID
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'started_at': datetime.utcnow().isoformat(),
            'track_count': 0,
            'last_activity': datetime.utcnow().isoformat()
        }
        
        # Store active session
        session_key = self._get_session_key(user_id)
        await cache_service.set_in_cache(
            session_key,
            session_data,
            expiration=self.SESSION_TIMEOUT
        )
        
        # Initialize empty queue
        queue_key = self._get_queue_key(user_id, session_id)
        await cache_service.set_in_cache(
            queue_key,
            {'tracks': [], 'session': session_data, 'current_mood': None, 'aggregated_features': {}},
            expiration=self.SESSION_TIMEOUT
        )
        
        print(f"🎧 Started live session {session_id} for user {user_id}")
        return session_id
    
    async def add_track_to_queue(
        self,
        user_id: str,
        session_id: str,
        track_data: Dict
    ) -> Dict:
        """
        Add track to live queue and recalculate aggregate features
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            track_data: Track dictionary with features and mood
            
        Returns:
            Updated queue analytics
        """
        queue_key = self._get_queue_key(user_id, session_id)
        session_key = self._get_session_key(user_id)
        
        # Get current queue
        queue_data = await cache_service.get_from_cache(queue_key)

        # ✅ Safeguard: handle missing or invalid queue_data gracefully
        if not isinstance(queue_data, dict):
            print(f"⚠️  Queue data not found or invalid for session {session_id}, creating fallback queue.")
            await self.start_session(user_id)
            queue_data = {'tracks': [], 'session': {}, 'current_mood': None, 'aggregated_features': {}}
        
        # Add track to queue
        tracks = queue_data.get('tracks', [])
        
        # Add timestamp
        track_entry = {
            **track_data,
            'played_at': datetime.utcnow().isoformat()
        }
        
        tracks.append(track_entry)
        
        # Limit queue size (keep most recent tracks)
        if len(tracks) > self.MAX_QUEUE_SIZE:
            tracks = tracks[-self.MAX_QUEUE_SIZE:]
        
        # Update queue
        queue_data['tracks'] = tracks
        
        # Recalculate aggregated features
        from .playlist_analyzer import playlist_analyzer
        aggregated = playlist_analyzer.aggregate_playlist_features(tracks)
        
        # Update session activity
        session_data = queue_data.get('session', {})
        session_data['last_activity'] = datetime.utcnow().isoformat()
        session_data['track_count'] = len(tracks)
        
        queue_data['session'] = session_data
        queue_data['aggregated_features'] = aggregated
        
        # Calculate current mood
        from .model_service import model_service
        mood_data = await model_service.predict_mood_from_features(
            aggregated,
            {'polarity': 0.0, 'subjectivity': 0.0},  # No lyrics for aggregate
            user_id=user_id
        )
        
        queue_data['current_mood'] = {
            'primary_mood': mood_data['primary_mood'],
            'all_moods': mood_data['all_moods'],
            'mood_scores': mood_data['mood_scores'],
            'confidence': mood_data['confidence']
        }
        
        # Save back to Redis
        await cache_service.set_in_cache(
            queue_key,
            queue_data,
            expiration=self.SESSION_TIMEOUT
        )
        
        # Update active session
        await cache_service.set_in_cache(
            session_key,
            session_data,
            expiration=self.SESSION_TIMEOUT
        )
        
        print(f"➕ Added track to queue: {len(tracks)} tracks, mood: {mood_data['primary_mood']}")
        
        return {
            'session_id': session_id,
            'track_count': len(tracks),
            'aggregated_features': aggregated,
            'current_mood': queue_data['current_mood'],
            'last_track': track_entry.get('name', 'Unknown')
        }
    
    async def get_current_queue(self, user_id: str, session_id: str) -> Optional[Dict]:
        """
        Get current live queue analytics
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Queue data with analytics
        """
        queue_key = self._get_queue_key(user_id, session_id)
        return await cache_service.get_from_cache(queue_key)
    
    async def get_active_session(self, user_id: str) -> Optional[Dict]:
        """
        Get user's active session if exists
        
        Args:
            user_id: User identifier
            
        Returns:
            Active session data or None
        """
        session_key = self._get_session_key(user_id)
        return await cache_service.get_from_cache(session_key)
    
    async def end_session(self, user_id: str, session_id: str) -> Dict:
        """
        End live session and save to MongoDB
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Final session analytics
        """
        queue_key = self._get_queue_key(user_id, session_id)
        queue_data = await cache_service.get_from_cache(queue_key)
        
        if not queue_data:
            return {'error': 'Session not found'}
        
        tracks = queue_data.get('tracks', [])
        
        if not tracks:
            return {'error': 'No tracks in session'}
        
        # Final analytics
        final_analytics = {
            'user_id': user_id,
            'session_id': session_id,
            'started_at': queue_data['session']['started_at'],
            'ended_at': datetime.utcnow().isoformat(),
            'track_count': len(tracks),
            'aggregated_features': queue_data.get('aggregated_features', {}),
            'final_mood': queue_data.get('current_mood', {}),
            'session_duration_minutes': self._calculate_duration(
                queue_data['session']['started_at'],
                datetime.utcnow().isoformat()
            )
        }
        
        # Save to MongoDB (via motor)
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            import os
            
            mongo_uri = os.getenv('MONGO_URI')
            if mongo_uri:
                client = AsyncIOMotorClient(mongo_uri)
                db = client['moodiq']
                collection = db['moodanalytics']
                
                await collection.insert_one(final_analytics)
                print(f"💾 Saved session analytics to MongoDB")
        except Exception as e:
            print(f"⚠️ Failed to save to MongoDB: {e}")
        
        # Clean up Redis
        await cache_service.delete_from_cache(queue_key)
        session_key = self._get_session_key(user_id)
        await cache_service.delete_from_cache(session_key)
        
        print(f"✅ Ended session {session_id}")
        
        return final_analytics
    
    @staticmethod
    def _calculate_duration(start_time: str, end_time: str) -> float:
        """Calculate duration in minutes"""
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            duration = (end - start).total_seconds() / 60.0
            return round(duration, 2)
        except:
            return 0.0
    
    async def check_and_auto_end_session(self, user_id: str) -> Optional[Dict]:
        """
        Check if session should be auto-ended due to inactivity
        
        Args:
            user_id: User identifier
            
        Returns:
            Final analytics if session was ended, None otherwise
        """
        session_data = await self.get_active_session(user_id)
        
        if not session_data:
            return None
        
        last_activity = datetime.fromisoformat(session_data['last_activity'].replace('Z', '+00:00'))
        now = datetime.utcnow()
        inactive_seconds = (now - last_activity).total_seconds()
        
        if inactive_seconds >= self.SESSION_TIMEOUT:
            # Auto-end session
            print(f"⏰ Auto-ending inactive session for user {user_id}")
            return await self.end_session(user_id, session_data['session_id'])
        
        return None


# Global instance
live_queue_service = LiveQueueService()
