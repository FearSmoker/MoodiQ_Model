"""
Updated Generate Router - COMPLETE VERSION
All endpoints preserved with proper mood mapping
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from services import music_service, model_service, lyrics_service, spotify_service, cache_service

router = APIRouter()


class GeneratePlaylistRequest(BaseModel):
    target_mood: str
    user_id: str
    seed_track_name: Optional[str] = None
    seed_artist_name: Optional[str] = None
    seed_track_id: Optional[str] = None
    limit: int = 20


class GenerateActivityRequest(BaseModel):
    activity: str
    user_id: str
    seed_track_name: Optional[str] = None
    seed_artist_name: Optional[str] = None
    seed_track_id: Optional[str] = None
    limit: int = 20


# ============================================
# HYBRID PLAYLIST GENERATION
# ============================================

@router.post("/playlist")
async def generate_mood_playlist(
    request: GeneratePlaylistRequest,
    authorization: str = Header(None)
):
    """
    Generate playlist for target mood (HYBRID APPROACH)
    NOW WITH MOOD MAPPING - Handles ANY mood request
    """
    try:
        # Map external mood to base mood
        base_mood = model_service.map_external_mood_to_base(request.target_mood)
        
        print(f"🎯 Generating playlist")
        print(f"   Requested mood: {request.target_mood}")
        print(f"   Base mood: {base_mood}")
        
        # Determine seed track
        seed_track_name = request.seed_track_name
        seed_artist_name = request.seed_artist_name
        
        # If Spotify track ID provided, get track info from Spotify
        if request.seed_track_id and authorization and authorization.startswith('Bearer '):
            try:
                access_token = authorization.replace('Bearer ', '')
                track_info = await spotify_service.get_track_info(request.seed_track_id, access_token)
                
                if track_info:
                    seed_track_name = track_info['name']
                    seed_artist_name = spotify_service.get_primary_artist_name(track_info)
                    print(f"✅ Got seed from Spotify: {seed_track_name} by {seed_artist_name}")
            except Exception as e:
                print(f"⚠️ Could not get Spotify track info: {e}")
        
        # If still no seed, use defaults based on BASE mood
        if not seed_track_name or not seed_artist_name:
            mood_seeds = {
                "Happy": ("Happy", "Pharrell Williams"),
                "Sad": ("Someone Like You", "Adele"),
                "Calm": ("Weightless", "Marconi Union"),
                "Energetic": ("Eye of the Tiger", "Survivor")
            }
            seed_track_name, seed_artist_name = mood_seeds[base_mood]
        
        print(f"🌱 Using seed: {seed_track_name} by {seed_artist_name}")
        
        # Get recommendations from Last.fm
        recommendations = await music_service.get_recommendations(
            seed_track_name=seed_track_name,
            seed_artist_name=seed_artist_name,
            target_mood=base_mood,  # Use base mood for filtering
            limit=request.limit * 2  # Get more for filtering
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations found. Try a different seed track."
            )
        
        print(f"✅ Found {len(recommendations)} candidate tracks from Last.fm")
        
        # Filter and analyze
        filtered_tracks = []
        
        for track in recommendations:
            try:
                # Already has features from get_recommendations
                features = track.get('features')
                
                if not features:
                    continue
                
                # Get lyrics sentiment
                lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                    track['name'],
                    track['artist']
                )
                
                # Predict mood
                mood_data = await model_service.predict_mood_from_features(
                    features,
                    lyrics_sentiment,
                    user_id=request.user_id,
                    genre=None
                )
                
                # Check if matches target BASE mood
                if mood_data['fused_mood'] == base_mood:
                    track['predicted_mood'] = mood_data['fused_mood']
                    track['confidence'] = mood_data['confidence']
                    track['mood_details'] = mood_data
                    filtered_tracks.append(track)
                
                if len(filtered_tracks) >= request.limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error processing track: {e}")
                continue
        
        if not filtered_tracks:
            print("⚠️ No exact mood matches, returning closest matches")
            filtered_tracks = recommendations[:request.limit]
        
        print(f"✅ Filtered to {len(filtered_tracks)} tracks matching {base_mood}")
        
        # Optimize flow if we have enough tracks
        if len(filtered_tracks) > 2:
            mood_profiles = {
                "Happy": {"valence": 0.8, "energy": 0.7, "danceability": 0.7},
                "Sad": {"valence": 0.2, "energy": 0.3, "danceability": 0.3},
                "Calm": {"valence": 0.5, "energy": 0.3, "danceability": 0.4},
                "Energetic": {"valence": 0.7, "energy": 0.9, "danceability": 0.8}
            }
            
            target_profile = mood_profiles[base_mood]
            
            optimization = model_service.optimize_flow_dp(
                filtered_tracks,
                target_profile,
                target_profile
            )
            
            ordered_tracks = [filtered_tracks[i] for i in optimization['optimizedOrder']]
            flow_score = optimization['flowScore']
        else:
            ordered_tracks = filtered_tracks
            flow_score = 1.0
        
        return {
            "requested_mood": request.target_mood,
            "target_mood": base_mood,
            "mood_mapped": request.target_mood.lower() != base_mood.lower(),
            "tracks": ordered_tracks,
            "total": len(ordered_tracks),
            "flow_score": flow_score,
            "seed_track": {
                "name": seed_track_name,
                "artist": seed_artist_name,
                "id": request.seed_track_id
            },
            "source": "lastfm_recommendations",
            "approach": "hybrid"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating mood playlist: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/activity")
async def generate_activity_playlist(
    request: GenerateActivityRequest,
    authorization: str = Header(None)
):
    """
    Generate playlist for specific activity (HYBRID APPROACH)
    """
    try:
        print(f"🏃 Generating playlist for activity: {request.activity}")
        
        activity_profiles = {
            "study": {
                "mood": "Calm",
                "seed": ("Clair de Lune", "Claude Debussy")
            },
            "workout": {
                "mood": "Energetic",
                "seed": ("Stronger", "Kanye West")
            },
            "gym": {
                "mood": "Energetic",
                "seed": ("Stronger", "Kanye West")
            },
            "party": {
                "mood": "Happy",
                "seed": ("Uptown Funk", "Mark Ronson")
            },
            "sleep": {
                "mood": "Calm",
                "seed": ("Weightless", "Marconi Union")
            },
            "meditation": {
                "mood": "Calm",
                "seed": ("Om Mani Padme Hum", "Imee Ooi")
            },
            "work": {
                "mood": "Calm",
                "seed": ("Lofi Hip Hop", "Various Artists")
            },
            "focus": {
                "mood": "Calm",
                "seed": ("Clair de Lune", "Claude Debussy")
            },
            "driving": {
                "mood": "Energetic",
                "seed": ("Life is a Highway", "Tom Cochrane")
            },
            "relax": {
                "mood": "Calm",
                "seed": ("Weightless", "Marconi Union")
            },
            "chill": {
                "mood": "Calm",
                "seed": ("Weightless", "Marconi Union")
            }
        }
        
        activity_lower = request.activity.lower()
        
        if activity_lower not in activity_profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown activity. Choose from: {', '.join(activity_profiles.keys())}"
            )
        
        profile = activity_profiles[activity_lower]
        
        if not request.seed_track_name:
            request.seed_track_name, request.seed_artist_name = profile['seed']
        
        mood_request = GeneratePlaylistRequest(
            target_mood=profile['mood'],
            user_id=request.user_id,
            seed_track_name=request.seed_track_name,
            seed_artist_name=request.seed_artist_name,
            seed_track_id=request.seed_track_id,
            limit=request.limit
        )
        
        result = await generate_mood_playlist(mood_request, authorization)
        
        result['activity'] = request.activity
        result['activity_profile'] = profile
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating activity playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# SPOTIFY-BASED GENERATION (HYBRID)
# ============================================

@router.post("/spotify/from-top-tracks")
async def generate_from_spotify_top_tracks(
    request: Dict[str, Any],
    authorization: str = Header(None)
):
    """
    Generate playlist based on user's Spotify top tracks (HYBRID)
    
    Uses:
    - Spotify API: Get user's top tracks
    - Last.fm: Get similar tracks for each top track
    - Multi-API: Audio features and mood prediction
    """
    try:
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        user_id = request.get('user_id')
        target_mood_raw = request.get('target_mood')
        limit = request.get('limit', 20)
        time_range = request.get('time_range', 'medium_term')  # short_term, medium_term, long_term
        
        # Map target mood to base mood if provided
        target_mood = None
        if target_mood_raw:
            target_mood = model_service.map_external_mood_to_base(target_mood_raw)
            print(f"🎯 Target mood: {target_mood_raw} → {target_mood}")
        
        print(f"🎵 Generating playlist from user's top tracks")
        
        # Get user's top tracks from Spotify
        top_tracks = await spotify_service.get_user_top_tracks(
            access_token,
            time_range=time_range,
            limit=5  # Use top 5 as seeds
        )
        
        if not top_tracks:
            raise HTTPException(status_code=404, detail="No top tracks found")
        
        print(f"✅ Got {len(top_tracks)} top tracks from Spotify")
        
        # Generate recommendations from each top track
        all_recommendations = []
        
        for top_track in top_tracks[:3]:  # Use top 3 as seeds
            track_name = top_track['name']
            artist_name = top_track['artists'][0]['name']
            
            print(f"🌱 Getting recommendations for: {track_name}")
            
            recommendations = await music_service.get_similar_tracks_lastfm(
                track_name,
                artist_name,
                limit=10
            )
            
            all_recommendations.extend(recommendations)
        
        # Remove duplicates
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            key = f"{rec['name']}:{rec['artist']}"
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        
        print(f"✅ Got {len(unique_recommendations)} unique recommendations")
        
        # Analyze and filter
        analyzed_tracks = []
        
        for track in unique_recommendations:
            try:
                # Get audio features
                features = await music_service.get_audio_features(
                    track['name'],
                    track['artist']
                )
                
                if not features:
                    continue
                
                # Get lyrics sentiment
                lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                    track['name'],
                    track['artist']
                )
                
                # Predict mood
                mood_data = await model_service.predict_mood_from_features(
                    features,
                    lyrics_sentiment,
                    user_id=user_id
                )
                
                # Filter by target mood if specified
                if target_mood and mood_data['fused_mood'] != target_mood:
                    continue
                
                track['features'] = features
                track['mood'] = mood_data['fused_mood']
                track['confidence'] = mood_data['confidence']
                track['mood_details'] = mood_data
                
                analyzed_tracks.append(track)
                
                if len(analyzed_tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error analyzing track: {e}")
                continue
        
        return {
            "source": "spotify_top_tracks",
            "top_tracks_used": [
                {"name": t['name'], "artists": [a['name'] for a in t['artists']]}
                for t in top_tracks[:3]
            ],
            "tracks": analyzed_tracks,
            "total": len(analyzed_tracks),
            "requested_mood": target_mood_raw,
            "target_mood": target_mood,
            "approach": "hybrid"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating from top tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spotify/from-recently-played")
async def generate_from_recently_played(
    request: Dict[str, Any],
    authorization: str = Header(None)
):
    """
    Generate playlist based on recently played tracks (HYBRID)
    
    Uses:
    - Spotify API: Get recently played tracks
    - Last.fm: Get similar tracks
    - Multi-API: Audio features and mood prediction
    """
    try:
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        user_id = request.get('user_id')
        target_mood_raw = request.get('target_mood')
        limit = request.get('limit', 20)
        
        # Map target mood
        target_mood = None
        if target_mood_raw:
            target_mood = model_service.map_external_mood_to_base(target_mood_raw)
        
        print(f"⏮️ Generating playlist from recently played tracks")
        
        # Get recently played from Spotify
        recent_tracks = await spotify_service.get_recently_played(
            access_token,
            limit=10
        )
        
        if not recent_tracks:
            raise HTTPException(status_code=404, detail="No recently played tracks found")
        
        print(f"✅ Got {len(recent_tracks)} recently played tracks")
        
        # Use most recent track as seed
        seed_track = recent_tracks[0]
        track_name = seed_track['name']
        artist_name = seed_track['artists'][0]['name']
        
        print(f"🌱 Using seed: {track_name} by {artist_name}")
        
        # Get recommendations
        recommendations = await music_service.get_recommendations(
            seed_track_name=track_name,
            seed_artist_name=artist_name,
            target_mood=target_mood,
            limit=limit * 2
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found")
        
        # Analyze and filter
        analyzed_tracks = []
        
        for track in recommendations:
            try:
                features = track.get('features')
                if not features:
                    continue
                
                lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                    track['name'],
                    track['artist']
                )
                
                mood_data = await model_service.predict_mood_from_features(
                    features,
                    lyrics_sentiment,
                    user_id=user_id
                )
                
                if target_mood and mood_data['fused_mood'] != target_mood:
                    continue
                
                track['mood'] = mood_data['fused_mood']
                track['confidence'] = mood_data['confidence']
                track['mood_details'] = mood_data
                
                analyzed_tracks.append(track)
                
                if len(analyzed_tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error analyzing track: {e}")
                continue
        
        return {
            "source": "recently_played",
            "seed_track": {
                "name": track_name,
                "artist": artist_name,
                "played_at": seed_track.get('played_at')
            },
            "tracks": analyzed_tracks,
            "total": len(analyzed_tracks),
            "requested_mood": target_mood_raw,
            "target_mood": target_mood,
            "approach": "hybrid"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating from recently played: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# DISCOVERY ENDPOINTS
# ============================================

@router.post("/discover")
async def discover_tracks(request: Dict[str, Any]):
    """
    Discover new tracks based on artist (Multi-API)
    
    Uses Last.fm similar artists to find new music
    """
    try:
        artist_name = request.get('artist_name')
        user_id = request.get('user_id')
        limit = request.get('limit', 20)
        
        if not artist_name:
            raise HTTPException(status_code=400, detail="Artist name is required")
        
        print(f"🔍 Discovering tracks similar to {artist_name}")
        
        similar_artists = await music_service.get_similar_artists_lastfm(
            artist_name,
            limit=10
        )
        
        if not similar_artists:
            return {
                "message": "No similar artists found",
                "tracks": []
            }
        
        discovered_tracks = []
        
        for artist in similar_artists[:5]:
            try:
                search_query = f"{artist['name']} official"
                tracks = await music_service.search_tracks(search_query, limit=3)
                
                for track in tracks:
                    features = await music_service.get_audio_features(
                        track['name'],
                        track['artists'][0] if track['artists'] else artist['name']
                    )
                    
                    if features:
                        lyrics_sentiment = {"polarity": 0.0, "subjectivity": 0.0}
                        
                        mood_data = await model_service.predict_mood_from_features(
                            features,
                            lyrics_sentiment,
                            user_id=user_id
                        )
                        
                        track['features'] = features
                        track['mood'] = mood_data['fused_mood']
                        track['confidence'] = mood_data['confidence']
                        track['similar_to_artist'] = artist_name
                        track['match_score'] = artist['match_score']
                        
                        discovered_tracks.append(track)
                
                if len(discovered_tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error discovering from artist {artist['name']}: {e}")
                continue
        
        discovered_tracks.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return {
            "seed_artist": artist_name,
            "discovered_tracks": discovered_tracks[:limit],
            "total": len(discovered_tracks),
            "similar_artists_explored": len(similar_artists)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error discovering tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/personalized")
async def generate_personalized_playlist(
    request: Dict[str, Any],
    authorization: str = Header(None)
):
    """
    Generate highly personalized playlist using user's feedback history
    """
    try:
        user_id = request.get('user_id')
        limit = request.get('limit', 30)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        print(f"🎯 Generating personalized playlist for user {user_id}")
        
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key)
        
        if not user_stats or user_stats.get('feedback_count', 0) < 5:
            return {
                "message": "Not enough feedback data for personalization. Please provide more feedback on tracks.",
                "tracks": [],
                "feedback_count": user_stats.get('feedback_count', 0) if user_stats else 0,
                "min_required": 5
            }
        
        mood_corrections = user_stats.get('mood_corrections', {})
        if mood_corrections:
            favorite_mood = max(mood_corrections, key=mood_corrections.get)
        else:
            favorite_mood = "Happy"
        
        print(f"📊 User's favorite mood: {favorite_mood}")
        
        mood_request = GeneratePlaylistRequest(
            target_mood=favorite_mood,
            user_id=user_id,
            limit=limit
        )
        
        result = await generate_mood_playlist(mood_request, authorization)
        
        result['personalized'] = True
        result['user_preferences'] = {
            "favorite_mood": favorite_mood,
            "mood_distribution": mood_corrections,
            "feedback_count": user_stats.get('feedback_count', 0)
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating personalized playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "playlist_generation",
        "approach": "hybrid",
        "features": [
            "Mood-based generation (Last.fm)",
            "Activity-based generation",
            "Discovery engine (Last.fm similar artists)",
            "Personalized playlists (user feedback)",
            "Spotify top tracks integration",
            "Spotify recently played integration",
            "Flow optimization (Dynamic Programming)",
            "External mood mapping (ANY mood → 4 base moods)"
        ],
        "supported_base_moods": model_service.MOOD_CLASSES,
        "mood_mapping_enabled": True,
        "data_sources": {
            "recommendations": "Last.fm",
            "user_data": "Spotify API",
            "audio_features": "AcousticBrainz + MusicBrainz",
            "mood_prediction": "ML Model + Lyrics"
        }
    }