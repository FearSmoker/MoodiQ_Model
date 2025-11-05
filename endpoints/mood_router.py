"""
Updated Mood Prediction Router - Hybrid Approach
Uses Spotify API for: metadata, playlists, currently playing
Uses Multi-API for: audio features, recommendations, mood analysis
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from services import lyrics_service, model_service, cache_service, music_service, spotify_service

router = APIRouter()


class TrackMoodRequest(BaseModel):
    """Request model for single track mood analysis"""
    track_name: str
    artist_name: str
    user_id: Optional[str] = None
    genre: Optional[str] = None


class SpotifyTrackMoodRequest(BaseModel):
    """Request model for Spotify track mood analysis"""
    track_id: str
    user_id: Optional[str] = None


class TrackMoodResponse(BaseModel):
    """Response model for track mood"""
    track_name: str
    artist_name: str
    mood: Dict[str, Any]
    features: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class PlaylistMoodRequest(BaseModel):
    """Request model for playlist mood analysis"""
    tracks: List[Dict[str, str]]  # List with 'name' and 'artist' keys
    user_id: Optional[str] = None


class SpotifyPlaylistMoodRequest(BaseModel):
    """Request model for Spotify playlist mood analysis"""
    playlist_id: str
    user_id: Optional[str] = None


class PlaylistMoodResponse(BaseModel):
    """Response model for playlist mood"""
    tracks: List[Dict[str, Any]]
    moodDistribution: Dict[str, float]
    overallMood: str


# ============================================
# ORIGINAL ENDPOINTS (Multi-API)
# ============================================

@router.post("/track", response_model=TrackMoodResponse)
async def get_track_mood(request: TrackMoodRequest):
    """
    Analyze mood for a single track using multi-API approach
    
    Pipeline:
    1. Search track on YTMusic
    2. Get MBID from MusicBrainz
    3. Fetch audio features from AcousticBrainz
    4. Get genre tags from Last.fm
    5. Fetch lyrics sentiment
    6. Predict mood using ML model
    """
    cache_key = f"track:mood:{request.track_name}:{request.artist_name}:{request.user_id or 'global'}"
    
    try:
        # Check cache
        cached_result = await cache_service.get_from_cache(cache_key)
        if cached_result:
            print(f"📦 Cache HIT for track mood")
            return cached_result
        
        print(f"🎵 Analyzing: {request.track_name} by {request.artist_name}")
        
        # 1. Get audio features (MusicBrainz → AcousticBrainz)
        audio_features = await music_service.get_audio_features(
            request.track_name,
            request.artist_name
        )
        
        if not audio_features:
            print("⚠️ Using default audio features")
            audio_features = music_service.get_default_features()
        
        # 2. Get genre tags from Last.fm
        tags = await music_service.get_lastfm_tags(
            request.track_name,
            request.artist_name
        )
        
        # Determine genre from tags
        genre = request.genre
        if not genre and tags:
            genre = tags[0]
        
        # 3. Get lyrics sentiment
        lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
            request.track_name,
            request.artist_name
        )
        
        # 4. Predict mood
        mood_data = await model_service.predict_mood_from_features(
            audio_features,
            lyrics_sentiment,
            user_id=request.user_id,
            track_id=None,
            genre=genre
        )
        
        result = {
            "track_name": request.track_name,
            "artist_name": request.artist_name,
            "mood": mood_data,
            "features": audio_features,
            "tags": tags[:10]
        }
        
        # Cache for 1 hour
        await cache_service.set_in_cache(cache_key, result, expiration=3600)
        
        print(f"✅ Mood analysis complete: {mood_data['fused_mood']}")
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing track mood: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/playlist", response_model=PlaylistMoodResponse)
async def get_playlist_mood(request: PlaylistMoodRequest):
    """
    Analyze mood for entire playlist (Multi-API)
    """
    try:
        print(f"🎵 Analyzing playlist with {len(request.tracks)} tracks")
        
        if not request.tracks:
            raise HTTPException(status_code=400, detail="No tracks provided")
        
        processed_tracks = []
        
        for idx, track_data in enumerate(request.tracks):
            try:
                track_name = track_data.get('name')
                artist_name = track_data.get('artist')
                
                if not track_name or not artist_name:
                    print(f"⚠️ Skipping track {idx}: missing name or artist")
                    continue
                
                cache_key = f"track:mood:{track_name}:{artist_name}:{request.user_id or 'global'}"
                cached_mood = await cache_service.get_from_cache(cache_key)
                
                if cached_mood:
                    print(f"📦 Cache HIT for track {idx + 1}/{len(request.tracks)}")
                    mood_data = cached_mood.get('mood')
                    audio_features = cached_mood.get('features')
                    tags = cached_mood.get('tags', [])
                else:
                    print(f"🔍 Processing track {idx + 1}/{len(request.tracks)}: {track_name}")
                    
                    audio_features = await music_service.get_audio_features(track_name, artist_name)
                    if not audio_features:
                        audio_features = music_service.get_default_features()
                    
                    tags = await music_service.get_lastfm_tags(track_name, artist_name)
                    genre = tags[0] if tags else None
                    
                    lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
                    
                    mood_data = await model_service.predict_mood_from_features(
                        audio_features,
                        lyrics_sentiment,
                        user_id=request.user_id,
                        track_id=None,
                        genre=genre
                    )
                    
                    track_cache = {
                        "track_name": track_name,
                        "artist_name": artist_name,
                        "mood": mood_data,
                        "features": audio_features,
                        "tags": tags
                    }
                    await cache_service.set_in_cache(cache_key, track_cache, expiration=3600)
                
                processed_tracks.append({
                    "name": track_name,
                    "artist": artist_name,
                    "features": audio_features,
                    "mood": mood_data.get('fused_mood', 'Unknown'),
                    "moodScore": mood_data.get('confidence', 0),
                    "moodDetails": mood_data,
                    "tags": tags[:5]
                })
                
            except Exception as track_error:
                print(f"❌ Error processing track {idx}: {track_error}")
                continue
        
        if not processed_tracks:
            raise HTTPException(status_code=500, detail="Failed to process any tracks")
        
        mood_stats = model_service.calculate_playlist_mood_distribution(processed_tracks)
        
        response = {
            "tracks": processed_tracks,
            "moodDistribution": mood_stats.get('distribution', {}),
            "overallMood": mood_stats.get('overall_mood', 'Mixed')
        }
        
        print(f"✅ Playlist analysis complete: {len(processed_tracks)} tracks")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Playlist mood analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# NEW SPOTIFY ENDPOINTS (Hybrid Approach)
# ============================================

@router.post("/spotify/track", response_model=Dict[str, Any])
async def get_spotify_track_mood(
    request: SpotifyTrackMoodRequest,
    authorization: str = Header(None)
):
    """
    Analyze mood for a Spotify track (HYBRID APPROACH)
    
    Uses:
    - Spotify API: Track metadata
    - Multi-API: Audio features, mood prediction
    
    Requires: Bearer token in Authorization header
    """
    try:
        # Extract access token
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        
        print(f"🎵 Analyzing Spotify track: {request.track_id}")
        
        # Get lyrics sentiment first
        # We'll get track name from Spotify, then fetch lyrics
        track_info = await spotify_service.get_track_info(request.track_id, access_token)
        
        if not track_info:
            raise HTTPException(status_code=404, detail="Track not found on Spotify")
        
        track_name = track_info['name']
        artist_name = spotify_service.get_primary_artist_name(track_info)
        
        lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
        
        # Use hybrid model service prediction
        mood_data = await model_service.predict_mood_from_spotify_track(
            track_id=request.track_id,
            access_token=access_token,
            lyrics_sentiment=lyrics_sentiment,
            user_id=request.user_id
        )
        
        print(f"✅ Spotify track mood analysis complete: {mood_data.get('fused_mood')}")
        return mood_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing Spotify track mood: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spotify/playlist", response_model=Dict[str, Any])
async def get_spotify_playlist_mood(
    request: SpotifyPlaylistMoodRequest,
    authorization: str = Header(None)
):
    """
    Analyze mood for entire Spotify playlist (HYBRID APPROACH)
    
    Uses:
    - Spotify API: Playlist tracks, metadata
    - Multi-API: Audio features, mood prediction
    
    Requires: Bearer token in Authorization header
    """
    try:
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        
        print(f"📂 Analyzing Spotify playlist: {request.playlist_id}")
        
        # Get playlist tracks from Spotify
        tracks = await spotify_service.get_playlist_tracks(request.playlist_id, access_token)
        
        if not tracks:
            raise HTTPException(status_code=404, detail="Playlist not found or empty")
        
        print(f"🎵 Found {len(tracks)} tracks in playlist")
        
        processed_tracks = []
        
        for idx, track in enumerate(tracks):
            try:
                track_id = track['id']
                track_name = track['name']
                artist_name = track['artists'][0]['name'] if track['artists'] else "Unknown"
                
                print(f"🔍 Processing track {idx + 1}/{len(tracks)}: {track_name}")
                
                # Check cache
                cache_key = f"spotify:track:mood:{track_id}:{request.user_id or 'global'}"
                cached_mood = await cache_service.get_from_cache(cache_key)
                
                if cached_mood:
                    print(f"📦 Cache HIT")
                    mood_data = cached_mood
                else:
                    # Get lyrics sentiment
                    lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
                    
                    # Predict mood using hybrid approach
                    mood_data = await model_service.predict_mood_from_spotify_track(
                        track_id=track_id,
                        access_token=access_token,
                        lyrics_sentiment=lyrics_sentiment,
                        user_id=request.user_id
                    )
                    
                    # Cache for 1 hour
                    await cache_service.set_in_cache(cache_key, mood_data, expiration=3600)
                
                # Build track response
                processed_tracks.append({
                    "id": track_id,
                    "name": track_name,
                    "artist": artist_name,
                    "album": track['album']['name'],
                    "duration_ms": track['duration_ms'],
                    "popularity": track['popularity'],
                    "external_url": track['external_url'],
                    "images": track['album']['images'],
                    "mood": mood_data.get('fused_mood', 'Unknown'),
                    "moodScore": mood_data.get('confidence', 0),
                    "moodDetails": mood_data,
                    "features": mood_data.get('track_info', {})
                })
                
            except Exception as track_error:
                print(f"⚠️ Error processing track {idx}: {track_error}")
                continue
        
        if not processed_tracks:
            raise HTTPException(status_code=500, detail="Failed to process any tracks")
        
        # Calculate mood distribution
        mood_stats = model_service.calculate_playlist_mood_distribution(processed_tracks)
        
        response = {
            "playlist_id": request.playlist_id,
            "tracks": processed_tracks,
            "total_tracks": len(processed_tracks),
            "moodDistribution": mood_stats.get('distribution', {}),
            "overallMood": mood_stats.get('overall_mood', 'Mixed'),
            "mood_diversity": mood_stats.get('mood_diversity', 0),
            "dominant_percentage": mood_stats.get('dominant_percentage', 0)
        }
        
        print(f"✅ Spotify playlist analysis complete: {len(processed_tracks)} tracks")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Spotify playlist mood analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spotify/currently-playing")
async def get_currently_playing_mood(
    authorization: str = Header(None),
    user_id: Optional[str] = None
):
    """
    Analyze mood of currently playing track (HYBRID APPROACH)
    
    Uses:
    - Spotify API: Currently playing track info
    - Multi-API: Audio features, mood prediction
    
    Requires: Bearer token in Authorization header
    """
    try:
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        
        print(f"🎧 Fetching currently playing track...")
        
        # Get currently playing from Spotify
        current = await spotify_service.get_currently_playing(access_token)
        
        if not current or not current.get('is_playing'):
            return {
                "is_playing": False,
                "message": "No track currently playing"
            }
        
        track = current['track']
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name'] if track['artists'] else "Unknown"
        
        print(f"🎵 Currently playing: {track_name} by {artist_name}")
        
        # Check cache
        cache_key = f"spotify:track:mood:{track_id}:{user_id or 'global'}"
        cached_mood = await cache_service.get_from_cache(cache_key)
        
        if cached_mood:
            print(f"📦 Cache HIT")
            mood_data = cached_mood
        else:
            # Get lyrics sentiment
            lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
            
            # Predict mood
            mood_data = await model_service.predict_mood_from_spotify_track(
                track_id=track_id,
                access_token=access_token,
                lyrics_sentiment=lyrics_sentiment,
                user_id=user_id
            )
            
            # Cache for 1 hour
            await cache_service.set_in_cache(cache_key, mood_data, expiration=3600)
        
        # Combine with playback info
        response = {
            "is_playing": True,
            "track": {
                "id": track_id,
                "name": track_name,
                "artists": [a['name'] for a in track['artists']],
                "album": track['album']['name'],
                "duration_ms": track['duration_ms'],
                "popularity": track['popularity'],
                "external_url": track['external_url'],
                "images": track['album']['images']
            },
            "device": current.get('device'),
            "progress_ms": current.get('progress_ms'),
            "shuffle_state": current.get('shuffle_state'),
            "repeat_state": current.get('repeat_state'),
            "mood_analysis": mood_data,
            "timestamp": current.get('timestamp')
        }
        
        print(f"✅ Currently playing mood analysis complete")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing currently playing track: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spotify/playlists")
async def get_user_playlists(authorization: str = Header(None)):
    """
    Get user's Spotify playlists
    
    Uses: Spotify API only
    Requires: Bearer token in Authorization header
    """
    try:
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        
        playlists = await spotify_service.get_user_playlists(access_token)
        
        return {
            "playlists": playlists,
            "total": len(playlists)
        }
        
    except Exception as e:
        print(f"❌ Error getting user playlists: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# UTILITY ENDPOINTS
# ============================================

@router.post("/search-and-analyze")
async def search_and_analyze(request: Dict[str, Any]):
    """
    Search for a track and analyze its mood (Multi-API)
    """
    try:
        query = request.get('query')
        user_id = request.get('user_id')
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        tracks = await music_service.search_tracks(query, limit=5)
        
        if not tracks:
            return {
                "message": "No tracks found",
                "results": []
            }
        
        first_track = tracks[0]
        
        mood_request = TrackMoodRequest(
            track_name=first_track['name'],
            artist_name=first_track['artists'][0] if first_track['artists'] else "Unknown",
            user_id=user_id
        )
        
        mood_result = await get_track_mood(mood_request)
        
        return {
            "search_results": tracks,
            "analyzed_track": mood_result,
            "total_results": len(tracks)
        }
        
    except Exception as e:
        print(f"❌ Search and analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-analyze")
async def batch_analyze_tracks(request: Dict[str, Any]):
    """
    Batch analyze multiple tracks efficiently
    """
    try:
        tracks = request.get('tracks', [])
        user_id = request.get('user_id')
        
        if not tracks:
            raise HTTPException(status_code=400, detail="Tracks list is required")
        
        print(f"🔄 Batch analyzing {len(tracks)} tracks")
        
        playlist_request = PlaylistMoodRequest(
            tracks=tracks,
            user_id=user_id
        )
        
        result = await get_playlist_mood(playlist_request)
        
        return {
            "tracks": result["tracks"],
            "total_analyzed": len(result["tracks"]),
            "mood_distribution": result["moodDistribution"],
            "overall_mood": result["overallMood"]
        }
        
    except Exception as e:
        print(f"❌ Batch analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model_service.mood_model is not None
    cache_connected = cache_service.is_connected()
    ytmusic_available = music_service.ytmusic is not None
    lastfm_configured = music_service.LASTFM_API_KEY is not None
    spotify_configured = spotify_service.sp_server is not None or True  # Can use user tokens
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ml_model": {
                "loaded": model_loaded,
                "status": "active" if model_loaded else "fallback"
            },
            "cache": {
                "connected": cache_connected
            },
            "spotify": {
                "configured": spotify_configured,
                "status": "hybrid_mode"
            },
            "ytmusic": {
                "available": ytmusic_available
            },
            "lastfm": {
                "configured": lastfm_configured
            },
            "musicbrainz": {
                "available": True
            },
            "acousticbrainz": {
                "available": True
            }
        },
        "api_stack": {
            "metadata": "Spotify API (user OAuth)",
            "audio_features": "MusicBrainz + AcousticBrainz",
            "recommendations": "Last.fm",
            "track_search": "YTMusicAPI",
            "lyrics": "Genius API"
        }
    }