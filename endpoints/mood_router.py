from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from services import spotify_service, lyrics_service, model_service, cache_service

router = APIRouter()


class TrackInfo(BaseModel):
    id: str
    name: str
    artists: List[str]
    album: Optional[str] = None
    duration_ms: Optional[int] = None
    preview_url: Optional[str] = None


class AudioFeature(BaseModel):
    """Audio features from Spotify API"""
    id: str
    valence: float
    energy: float
    danceability: float
    acousticness: float
    instrumentalness: float
    speechiness: float
    tempo: float
    loudness: float
    liveness: float
    key: int
    mode: int
    time_signature: int


class PlaylistMoodRequest(BaseModel):
    """Request model matching backend's POST /api/playlists/mood"""
    track_ids: List[str]
    audio_features: List[Dict[str, Any]]
    access_token: Optional[str] = None  # Make optional
    user_id: Optional[str] = None


class TrackMoodRequest(BaseModel):
    """Single track mood analysis"""
    track_id: str
    name: str
    artist: str
    access_token: Optional[str] = None  # Make optional
    user_id: Optional[str] = None
    genre: Optional[str] = None


class TrackMoodResponse(BaseModel):
    """Response model for track mood"""
    track_id: str
    name: str
    artists: List[str]
    mood: Dict[str, Any]
    features: Optional[Dict[str, Any]] = None


class PlaylistMoodResponse(BaseModel):
    """Response model matching backend expectations"""
    tracks: List[Dict[str, Any]]
    moodDistribution: Dict[str, float]
    overallMood: str


def validate_access_token(token: Optional[str]) -> Optional[str]:
    """
    Validate access token and return None if it's a placeholder/invalid.
    This allows the service to fall back to server credentials.
    """
    if not token:
        return None
    
    # Check for common placeholder values
    placeholders = [
        "YOUR_SPOTIFY_TOKEN",
        "your_spotify_token", 
        "YOUR_TOKEN",
        "your_token",
        "TOKEN",
        "token",
        "test_token",
        "dummy_token"
    ]
    
    if token.lower() in [p.lower() for p in placeholders]:
        print(f"⚠️  Invalid/placeholder token detected: '{token}' - using server credentials")
        return None
    
    # If token is too short, it's probably invalid
    if len(token) < 50:
        print(f"⚠️  Token too short ({len(token)} chars) - using server credentials")
        return None
    
    return token


@router.post("/track", response_model=TrackMoodResponse)
async def get_track_mood(request: TrackMoodRequest):
    """
    Analyze mood for a single track.
    Matches backend's single track analysis flow.
    """
    # Validate and clean the access token
    clean_token = validate_access_token(request.access_token)
    
    cache_key = f"track:mood:{request.track_id}:{request.user_id or 'global'}"
    
    try:
        # Check cache first
        cached_result = await cache_service.get_from_cache(cache_key)
        if cached_result:
            print(f"📦 Cache HIT for track {request.track_id}")
            return cached_result
        
        print(f"🔍 Cache MISS for track {request.track_id}")
        
        # 1. Get Audio Features
        # Use clean_token (None if invalid) - will fall back to server credentials
        features_list = await spotify_service.get_audio_features(
            [request.track_id], 
            clean_token
        )
        
        if not features_list or not features_list[0]:
            # Instead of 404, return a response with rule-based prediction
            print(f"⚠️  Track features not available, using rule-based prediction")
            
            # Use default features for rule-based prediction
            default_features = {
                'valence': 0.5,
                'energy': 0.5,
                'danceability': 0.5,
                'acousticness': 0.5,
                'instrumentalness': 0.0,
                'speechiness': 0.05,
                'tempo': 120.0,
                'loudness': -10.0,
                'liveness': 0.1,
                'key': 0,
                'mode': 1,
                'time_signature': 4
            }
            
            # Get lyrics sentiment (this might still work)
            lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                request.name, 
                request.artist
            )
            
            # Predict mood with default features
            mood_data = await model_service.predict_mood_from_features(
                default_features, 
                lyrics_sentiment,
                user_id=request.user_id,
                track_id=request.track_id,
                genre=request.genre
            )
            
            result = {
                "track_id": request.track_id,
                "name": request.name,
                "artists": [request.artist],
                "mood": {
                    **mood_data,
                    "warning": "Used default features - actual track features unavailable"
                },
                "features": None
            }
            
            # Cache for shorter time (1 hour) since features weren't available
            await cache_service.set_in_cache(cache_key, result, expiration=3600)
            
            return result
        
        audio_features = features_list[0]
        
        # 2. Get Lyrics Sentiment
        lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
            request.name, 
            request.artist
        )
        
        # 3. Predict Mood with user context and genre
        mood_data = await model_service.predict_mood_from_features(
            audio_features, 
            lyrics_sentiment,
            user_id=request.user_id,
            track_id=request.track_id,
            genre=request.genre
        )
        
        result = {
            "track_id": request.track_id,
            "name": request.name,
            "artists": [request.artist],
            "mood": mood_data,
            "features": audio_features
        }
        
        # Cache for 1 hour
        await cache_service.set_in_cache(cache_key, result, expiration=3600)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing track mood: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/playlist", response_model=PlaylistMoodResponse)
async def get_playlist_mood(request: PlaylistMoodRequest):
    """
    Analyze mood for entire playlist.
    This matches the backend's POST /api/playlists/mood endpoint.
    
    Backend sends:
    - track_ids: List of Spotify track IDs
    - audio_features: Already fetched from Spotify
    - access_token: User's Spotify token (optional)
    - user_id: User ID for personalization
    """
    # Validate and clean the access token
    clean_token = validate_access_token(request.access_token)
    
    try:
        print(f"🎵 Analyzing playlist with {len(request.track_ids)} tracks")
        
        if not request.track_ids or len(request.track_ids) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No tracks provided for analysis"
            )
        
        # Get Spotify client - use clean_token (None if invalid)
        try:
            sp = spotify_service.get_spotify_client(clean_token)
        except Exception as e:
            print(f"⚠️  Failed to get Spotify client with user token: {e}")
            print("   Falling back to server credentials...")
            sp = spotify_service.get_spotify_client(None)
        
        # Fetch track info in batches (Spotify allows max 50 per request)
        all_tracks_info = []
        for i in range(0, len(request.track_ids), 50):
            batch_ids = request.track_ids[i:i+50]
            try:
                tracks_response = sp.tracks(batch_ids)
                all_tracks_info.extend(tracks_response['tracks'])
            except Exception as e:
                print(f"⚠️  Error fetching track info batch {i//50 + 1}: {e}")
                # If this fails, we'll use track_ids as fallback below
                continue
        
        # Create a map for quick lookup
        track_info_map = {t['id']: t for t in all_tracks_info if t}
        
        processed_tracks = []
        
        # Process each track
        for idx, track_id in enumerate(request.track_ids):
            try:
                # Get corresponding audio features
                if idx >= len(request.audio_features):
                    print(f"⚠️  Missing audio features for track {track_id}")
                    continue
                
                audio_features = request.audio_features[idx]
                
                if not audio_features:
                    print(f"⚠️  Audio features are None for track {track_id}")
                    continue
                
                # Verify track ID matches (with fallback)
                features_id = audio_features.get('id')
                if features_id and features_id != track_id:
                    print(f"⚠️  Audio features ID mismatch: expected {track_id}, got {features_id}")
                    # Try to find matching features
                    matching_features = next(
                        (f for f in request.audio_features if f and f.get('id') == track_id),
                        None
                    )
                    if matching_features:
                        audio_features = matching_features
                        print(f"✅ Found matching features for {track_id}")
                    else:
                        print(f"⚠️  No matching features found, skipping {track_id}")
                        continue
                
                # Get track info (with fallback to basic info)
                track_info = track_info_map.get(track_id)
                
                if track_info:
                    name = track_info['name']
                    artists = [artist['name'] for artist in track_info['artists']]
                    artist_str = artists[0] if artists else "Unknown"
                    album = track_info['album']['name']
                    duration_ms = track_info.get('duration_ms')
                    preview_url = track_info.get('preview_url')
                    
                    # Get genre from artist info (if available)
                    genre = None
                    try:
                        if track_info['artists']:
                            artist_id = track_info['artists'][0]['id']
                            artist_info = sp.artist(artist_id)
                            if artist_info.get('genres'):
                                genre = artist_info['genres'][0]
                    except Exception:
                        pass
                else:
                    # Fallback: use basic info from track_id
                    print(f"⚠️  Track info not available for {track_id}, using basic info")
                    name = f"Track {idx + 1}"
                    artists = ["Unknown Artist"]
                    artist_str = "Unknown Artist"
                    album = "Unknown Album"
                    duration_ms = None
                    preview_url = None
                    genre = None
                
                # Check user-specific cache
                cache_key = f"track:mood:{track_id}:{request.user_id or 'global'}"
                cached_mood = await cache_service.get_from_cache(cache_key)
                
                if cached_mood:
                    print(f"📦 Cache HIT for track {track_id} in playlist")
                    mood_data = cached_mood.get('mood')
                else:
                    print(f"🔍 Cache MISS for track {track_id} in playlist")
                    
                    # Get Lyrics Sentiment
                    lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                        name, 
                        artist_str
                    )
                    
                    # Predict Mood with personalization
                    mood_data = await model_service.predict_mood_from_features(
                        audio_features, 
                        lyrics_sentiment,
                        user_id=request.user_id,
                        track_id=track_id,
                        genre=genre
                    )
                    
                    # Cache the result
                    track_cache_data = {
                        "track_id": track_id,
                        "name": name,
                        "artists": artists,
                        "mood": mood_data,
                        "features": audio_features
                    }
                    await cache_service.set_in_cache(
                        cache_key, 
                        track_cache_data, 
                        expiration=3600
                    )
                
                # Build response matching backend expectations
                processed_tracks.append({
                    "id": track_id,
                    "name": name,
                    "artists": artists,
                    "album": album,
                    "duration_ms": duration_ms,
                    "preview_url": preview_url,
                    "features": audio_features,
                    "mood": mood_data.get('fused_mood', 'Unknown'),
                    "moodScore": mood_data.get('confidence', 0),
                    "moodDetails": mood_data  # Full mood data for frontend
                })
                
            except Exception as track_error:
                print(f"❌ Error processing track {track_id}: {track_error}")
                import traceback
                traceback.print_exc()
                # Continue with other tracks
                continue
        
        if not processed_tracks:
            raise HTTPException(
                status_code=500,
                detail="Failed to process any tracks - check if audio features are provided correctly"
            )
        
        # Calculate mood distribution
        mood_stats = model_service.calculate_playlist_mood_distribution(processed_tracks)
        
        response = {
            "tracks": processed_tracks,
            "moodDistribution": mood_stats.get('distribution', {}),
            "overallMood": mood_stats.get('overall_mood', 'Mixed')
        }
        
        print(f"✅ Successfully analyzed {len(processed_tracks)}/{len(request.track_ids)} tracks")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Playlist mood analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during playlist analysis: {str(e)}"
        )


@router.post("/gaps")
async def detect_mood_gaps(request: Dict[str, Any]):
    """
    Detect jarring mood transitions (gaps) in a playlist.
    Helps identify where songs should be added for smoother flow.
    """
    try:
        tracks = request.get('tracks', [])
        threshold = request.get('threshold', 1.5)
        
        if not tracks:
            raise HTTPException(status_code=400, detail="No tracks provided")
        
        gaps = model_service.detect_mood_gaps(tracks, threshold)
        
        return {
            "gaps": gaps,
            "total_gaps": len(gaps),
            "needs_optimization": len(gaps) > len(tracks) * 0.2  # >20% gaps
        }
        
    except Exception as e:
        print(f"❌ Error detecting gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model_service.mood_model is not None
    cache_connected = cache_service.is_connected()
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "cache_connected": cache_connected,
        "service": "mood_prediction"
    }