"""
MongoDB Database-Driven Recommendation Service
===============================================
Uses MongoDB vector search on databasesongs collection
"""

import os
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np


class DatabaseRecommendationService:
    """
    MongoDB-based recommendation engine using vector similarity
    """
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize MongoDB connection"""
        if self._initialized:
            return
        
        mongo_uri = os.getenv('MONGO_URI')
        if not mongo_uri:
            print("⚠️ MONGO_URI not set, database recommendations disabled")
            return
        
        try:
            self.client = AsyncIOMotorClient(mongo_uri)
            self.db = self.client['moodiq']
            self.collection = self.db['databasesongs']
            
            # Test connection
            await self.client.admin.command('ping')
            
            # Check if collection exists
            count = await self.collection.count_documents({})
            print(f"✅ Connected to MongoDB: {count:,} songs available")
            
            self._initialized = True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            self._initialized = False
    
    def calculate_feature_vector(self, features: Dict) -> List[float]:
        """
        Convert audio features to normalized vector for similarity search
        
        Args:
            features: Audio features dictionary
            
        Returns:
            Normalized feature vector
        """
        # Feature order (must match database)
        feature_names = [
            'valence', 'energy', 'danceability', 'acousticness',
            'instrumentalness', 'speechiness', 'tempo', 'loudness',
            'liveness'  # Exclude key, mode, time_signature from similarity
        ]
        
        vector = []
        for feature_name in feature_names:
            value = features.get(feature_name, 0.5)
            
            # Normalize specific features
            if feature_name == 'tempo':
                value = min(max((value - 40) / 160, 0), 1)  # Normalize 40-200 BPM
            elif feature_name == 'loudness':
                value = min(max((value + 60) / 60, 0), 1)  # Normalize -60 to 0 dB
            
            vector.append(float(value))
        
        # L2 normalize
        vector_array = np.array(vector)
        norm = np.linalg.norm(vector_array)
        if norm > 0:
            vector_array = vector_array / norm
        
        return vector_array.tolist()
    
    def calculate_cosine_similarity(
        self,
        vector1: List[float],
        features2: Dict
    ) -> float:
        """
        Calculate cosine similarity between feature vector and track features
        
        Args:
            vector1: Normalized feature vector
            features2: Track features dictionary
            
        Returns:
            Similarity score (0-1)
        """
        vector2 = self.calculate_feature_vector(features2)
        
        dot_product = np.dot(vector1, vector2)
        
        # Cosine similarity is already in [-1, 1], map to [0, 1]
        similarity = (dot_product + 1) / 2
        
        return float(similarity)
    
    async def find_similar_tracks(
        self,
        target_features: Dict,
        target_moods: Optional[List[str]] = None,
        limit: int = 50,
        min_similarity: float = 0.75,
        exclude_track_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Find similar tracks using MongoDB vector search
        
        Args:
            target_features: Target audio features
            target_moods: Optional mood filters
            limit: Maximum results (10-50)
            min_similarity: Minimum similarity threshold
            exclude_track_ids: Track IDs to exclude
            
        Returns:
            List of similar tracks with similarity scores
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._initialized:
            return []
        
        print(f"🔍 Finding similar tracks in database...")
        print(f"   Target moods: {target_moods}")
        print(f"   Limit: {limit}, Min similarity: {min_similarity}")
        
        # Calculate target feature vector
        target_vector = self.calculate_feature_vector(target_features)
        
        # Build MongoDB query
        query = {}
        
        # Filter by moods if specified
        if target_moods:
            query['$or'] = [
                {'moods.primary_mood': {'$in': target_moods}},
                {'moods.all_moods': {'$in': target_moods}}
            ]
        
        # Exclude specific tracks
        if exclude_track_ids:
            query['track_id'] = {'$nin': exclude_track_ids}
        
        # Get candidate tracks (retrieve more for better filtering)
        candidates_limit = limit * 5
        
        try:
            cursor = self.collection.find(query).limit(candidates_limit)
            candidates = await cursor.to_list(length=candidates_limit)
            
            print(f"   Retrieved {len(candidates)} candidates from database")
            
            if not candidates:
                print("   ⚠️ No candidates found")
                return []
            
            # Calculate similarity for each candidate
            scored_tracks = []
            
            for track in candidates:
                features = track.get('features', {})
                
                if not features:
                    continue
                
                similarity = self.calculate_cosine_similarity(target_vector, features)
                
                if similarity >= min_similarity:
                    scored_tracks.append({
                        'track_id': track['track_id'],
                        'track_name': track['track_name'],
                        'artist_name': track['artist_name'],
                        'features': features,
                        'moods': track.get('moods', {}),
                        'metadata': track.get('metadata', {}),
                        'similarity_score': round(similarity, 4),
                        'source': 'database'
                    })
            
            # Sort by similarity (descending)
            scored_tracks.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Limit results
            results = scored_tracks[:limit]
            
            print(f"   ✅ Found {len(results)} similar tracks")
            
            if results:
                print(f"   Top match: {results[0]['track_name']} ({results[0]['similarity_score']:.3f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Database search error: {e}")
            return []
    
    async def get_recommendations_for_playlist(
        self,
        playlist_features: Dict,
        playlist_moods: List[str],
        limit: int = 50,
        diversity_factor: float = 0.2
    ) -> List[Dict]:
        """
        Get recommendations based on aggregated playlist features
        
        Args:
            playlist_features: Aggregated playlist audio features
            playlist_moods: Detected playlist moods
            limit: Number of recommendations
            diversity_factor: How much diversity to introduce (0-1)
            
        Returns:
            List of recommended tracks
        """
        print(f"🎯 Generating database recommendations for playlist")
        
        # Find highly similar tracks
        core_recommendations = await self.find_similar_tracks(
            playlist_features,
            target_moods=playlist_moods,
            limit=int(limit * (1 - diversity_factor)),
            min_similarity=0.80
        )
        
        # Find diverse tracks (lower similarity but same moods)
        diverse_recommendations = await self.find_similar_tracks(
            playlist_features,
            target_moods=playlist_moods,
            limit=int(limit * diversity_factor),
            min_similarity=0.65,
            exclude_track_ids=[t['track_id'] for t in core_recommendations]
        )
        
        # Combine and shuffle
        all_recommendations = core_recommendations + diverse_recommendations
        
        # Interleave for better variety
        final_recommendations = []
        for i in range(max(len(core_recommendations), len(diverse_recommendations))):
            if i < len(core_recommendations):
                final_recommendations.append(core_recommendations[i])
            if i < len(diverse_recommendations):
                final_recommendations.append(diverse_recommendations[i])
        
        return final_recommendations[:limit]
    
    async def get_mood_based_recommendations(
        self,
        mood: str,
        limit: int = 50,
        feature_preferences: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get recommendations based purely on mood
        
        Args:
            mood: Target mood
            limit: Number of recommendations
            feature_preferences: Optional feature preferences
            
        Returns:
            List of tracks matching mood
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._initialized:
            return []
        
        print(f"🎭 Finding tracks for mood: {mood}")
        
        # Query for mood
        query = {
            '$or': [
                {'moods.primary_mood': mood},
                {'moods.all_moods': mood}
            ]
        }
        
        try:
            # Get tracks with specified mood
            cursor = self.collection.find(query).limit(limit * 2)
            candidates = await cursor.to_list(length=limit * 2)
            
            print(f"   Found {len(candidates)} candidates")
            
            if not candidates:
                return []
            
            # If feature preferences provided, rank by similarity
            if feature_preferences:
                target_vector = self.calculate_feature_vector(feature_preferences)
                
                for track in candidates:
                    features = track.get('features', {})
                    similarity = self.calculate_cosine_similarity(target_vector, features)
                    track['preference_score'] = similarity
                
                candidates.sort(key=lambda x: x.get('preference_score', 0), reverse=True)
            
            # Format results
            results = []
            for track in candidates[:limit]:
                results.append({
                    'track_id': track['track_id'],
                    'track_name': track['track_name'],
                    'artist_name': track['artist_name'],
                    'features': track.get('features', {}),
                    'moods': track.get('moods', {}),
                    'metadata': track.get('metadata', {}),
                    'source': 'database'
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Mood-based search error: {e}")
            return []
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✅ MongoDB connection closed")


# Global instance
db_recommendation_service = DatabaseRecommendationService()