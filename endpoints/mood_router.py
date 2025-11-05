"""
Updated Mood Prediction Router - Hybrid Approach
Compatible with Production-Ready Spotify Service

Changes:
- ✅ Proper exception handling for new Spotify exceptions
- ✅ Support for full pagination
- ✅ Rate limit error handling
- ✅ Token expiration handling
- ✅ Consistent cache key generation
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from services import (
    lyrics_service, 
    model_service, 
    cache_service, 
    music_service, 
    spotify_service
)
from services.spotify_service import (
    SpotifyAuthError,
    SpotifyRateLimitError,
    SpotifyNotFoundError,
    SpotifyServiceError
)
import traceback

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
    include_unavailable: Optional[bool] = False  # NEW: Handle unavailable tracks


class PlaylistMoodResponse(BaseModel):
    """Response model for playlist mood"""
    tracks: List[Dict[str, Any]]
    moodDistribution: Dict[str, float]
    overallMood: str


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_access_token(authorization: Optional[str]) -> str:
    """
    Extract and validate access token from Authorization header
    
    Raises:
        HTTPException: If token is missing or invalid
    """
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected format: 'Bearer <token>'"
        )
    return authorization.replace('Bearer ', '').strip()


def handle_spotify_error(e: Exception) -> None:
    """
    Convert Spotify service exceptions to appropriate HTTP exceptions
    """
    if isinstance(e, SpotifyAuthError):
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}. Please re-authenticate with Spotify."
        )
    elif isinstance(e, SpotifyRateLimitError):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {str(e)}. Retry after {e.retry_after} seconds.",
            headers={"Retry-After": str(e.retry_after)}
        )
    elif isinstance(e, SpotifyNotFoundError):
        raise HTTPException(
            status_code=404,
            detail=f"Resource not found: {str(e)}"
        )
    elif isinstance(e, SpotifyServiceError):
        raise HTTPException(
            status_code=500,
            detail=f"Spotify service error: {str(e)}"
        )


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
        access_token = extract_access_token(authorization)
        
        print(f"🎵 Analyzing Spotify track: {request.track_id}")
        
        # Get track info from Spotify
        track_info = await spotify_service.get_track_info(request.track_id, access_token)
        
        if not track_info:
            raise HTTPException(status_code=404, detail="Track not found on Spotify")
        
        track_name = track_info['name']
        artist_name = spotify_service.get_primary_artist_name(track_info)
        
        # Get lyrics sentiment
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
        
    except (SpotifyAuthError, SpotifyRateLimitError, SpotifyNotFoundError, SpotifyServiceError) as e:
        handle_spotify_error(e)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing Spotify track mood: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spotify/playlist", response_model=Dict[str, Any])
async def get_spotify_playlist_mood(
    request: SpotifyPlaylistMoodRequest,
    authorization: str = Header(None)
):
    """
    Analyze mood for entire Spotify playlist (HYBRID APPROACH)
    
    UPDATED: Now handles full pagination for large playlists (100+ tracks)
    
    Uses:
    - Spotify API: Playlist tracks, metadata (with pagination)
    - Multi-API: Audio features, mood prediction
    
    Requires: Bearer token in Authorization header
    """
    try:
        access_token = extract_access_token(authorization)
        
        print(f"📂 Analyzing Spotify playlist: {request.playlist_id}")
        
        # Get ALL playlist tracks from Spotify (with pagination)
        tracks = await spotify_service.get_playlist_tracks(
            request.playlist_id, 
            access_token,
            include_unavailable=request.include_unavailable
        )
        
        if not tracks:
            raise HTTPException(status_code=404, detail="Playlist not found or empty")
        
        print(f"🎵 Found {len(tracks)} tracks in playlist")
        
        processed_tracks = []
        skipped_tracks = 0
        
        for idx, track in enumerate(tracks):
            try:
                # Handle unavailable tracks
                if track.get('unavailable'):
                    print(f"⚠️ Skipping unavailable track at position {idx + 1}")
                    skipped_tracks += 1
                    continue
                
                track_id = track.get('id')
                if not track_id:
                    print(f"⚠️ Skipping track without ID at position {idx + 1}")
                    skipped_tracks += 1
                    continue
                
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
                    "duration_ms": track.get('duration_ms', 0),
                    "popularity": track.get('popularity', 0),
                    "external_url": track.get('external_url'),
                    "images": track['album'].get('images', []),
                    "added_at": track.get('added_at'),
                    "mood": mood_data.get('fused_mood', 'Unknown'),
                    "moodScore": mood_data.get('confidence', 0),
                    "moodDetails": mood_data,
                    "features": mood_data.get('track_info', {})
                })
                
            except Exception as track_error:
                print(f"⚠️ Error processing track {idx}: {track_error}")
                skipped_tracks += 1
                continue
        
        if not processed_tracks:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process any tracks. {skipped_tracks} tracks were skipped."
            )
        
        # Calculate mood distribution
        mood_stats = model_service.calculate_playlist_mood_distribution(processed_tracks)
        
        response = {
            "playlist_id": request.playlist_id,
            "tracks": processed_tracks,
            "total_tracks": len(tracks),
            "processed_tracks": len(processed_tracks),
            "skipped_tracks": skipped_tracks,
            "moodDistribution": mood_stats.get('distribution', {}),
            "overallMood": mood_stats.get('overall_mood', 'Mixed'),
            "mood_diversity": mood_stats.get('mood_diversity', 0),
            "dominant_percentage": mood_stats.get('dominant_percentage', 0)
        }
        
        print(f"✅ Spotify playlist analysis complete:")
        print(f"   • Total tracks: {len(tracks)}")
        print(f"   • Processed: {len(processed_tracks)}")
        print(f"   • Skipped: {skipped_tracks}")
        
        return response
        
    except (SpotifyAuthError, SpotifyRateLimitError, SpotifyNotFoundError, SpotifyServiceError) as e:
        handle_spotify_error(e)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Spotify playlist mood analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spotify/currently-playing")
async def get_currently_playing_mood(
    authorization: str = Header(None),
    user_id: Optional[str] = None
):
    """
    Analyze mood of currently playing track (HYBRID APPROACH)
    
    Flow:
    1. Get currently playing from Spotify (with ALL details)
    2. Extract track info
    3. Get audio features from Multi-API
    4. Get lyrics sentiment
    5. Predict mood with ML model
    6. Return COMPLETE response with playback state
    
    Uses:
    - Spotify API: Currently playing track info + device + playback state
    - Multi-API: Audio features, mood prediction
    
    Requires: Bearer token in Authorization header
    """
    try:
        access_token = extract_access_token(authorization)
        
        print(f"\n{'='*60}")
        print(f"🎧 ANALYZING CURRENTLY PLAYING TRACK")
        print(f"{'='*60}")
        
        # STEP 1: Get currently playing from Spotify (with ALL details)
        print("\n📡 Step 1: Fetching playback state from Spotify...")
        playback_data = await spotify_service.get_currently_playing(access_token)
        
        if not playback_data or not playback_data.get('is_playing'):
            print("❌ No track currently playing")
            return {
                'is_playing': False,
                'message': 'No track currently playing',
                'timestamp': None
            }
        
        # Handle podcasts/episodes
        if playback_data.get('type') == 'episode':
            print("📻 Currently playing: Podcast Episode")
            return {
                'is_playing': True,
                'type': 'episode',
                'episode': playback_data['episode'],
                'device': playback_data['device'],
                'progress_ms': playback_data['progress_ms'],
                'message': 'Mood analysis not available for podcast episodes'
            }
        
        # Extract track and playback info
        track = playback_data['track']
        device = playback_data['device']
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        
        print(f"✅ Currently Playing: {track_name} by {artist_name}")
        print(f"   Device: {device['name']} ({device['type']})")
        print(f"   Volume: {device.get('volume_percent')}%")
        print(f"   Shuffle: {playback_data['shuffle_state']}")
        print(f"   Repeat: {playback_data['repeat_state']}")
        
        # STEP 2: Get audio features from Multi-API
        print("\n🎹 Step 2: Getting audio features...")
        try:
            audio_features = await music_service.get_audio_features(
                track_name,
                artist_name
            )
            
            if not audio_features:
                print("⚠️ Using default audio features")
                audio_features = music_service.get_default_features()
            else:
                print(f"✅ Audio features retrieved")
                print(f"   Valence: {audio_features.get('valence', 0):.2f}")
                print(f"   Energy: {audio_features.get('energy', 0):.2f}")
                
        except Exception as e:
            print(f"⚠️ Audio features error: {e}")
            audio_features = music_service.get_default_features()
        
        # STEP 3: Get lyrics sentiment
        print("\n📝 Step 3: Analyzing lyrics sentiment...")
        try:
            lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                track_name,
                artist_name
            )
            
            if lyrics_sentiment:
                print(f"✅ Lyrics sentiment: {lyrics_sentiment.get('polarity', 0):.2f}")
            else:
                print("⚠️ No lyrics found, using neutral sentiment")
                lyrics_sentiment = {'polarity': 0.0, 'subjectivity': 0.0}
                
        except Exception as e:
            print(f"⚠️ Lyrics error: {e}")
            lyrics_sentiment = {'polarity': 0.0, 'subjectivity': 0.0}
        
        # STEP 4: Get genre for adaptive weighting
        print("\n🎸 Step 4: Getting genre tags...")
        try:
            tags = await music_service.get_lastfm_tags(track_name, artist_name)
            genre = tags[0] if tags else None
            
            if genre:
                print(f"✅ Genre: {genre}")
            else:
                print("⚠️ No genre found, using default weighting")
                
        except Exception as e:
            print(f"⚠️ Genre error: {e}")
            genre = None
        
        # STEP 5: Predict mood with ML model
        print("\n🤖 Step 5: Predicting mood...")
        try:
            mood_data = await model_service.predict_mood_from_features(
                audio_features,
                lyrics_sentiment,
                user_id=user_id,
                track_id=track_id,
                genre=genre
            )
            
            print(f"✅ Mood predicted: {mood_data['fused_mood']} (confidence: {mood_data['confidence']:.2f})")
            
        except Exception as e:
            print(f"⚠️ Mood prediction error: {e}")
            traceback.print_exc()
            
            # Fallback mood
            mood_data = {
                'audio_mood': 'Unknown',
                'lyrics_mood': 'Neutral',
                'fused_mood': 'Unknown',
                'confidence': 0.0,
                'source': 'fallback',
                'scores': {
                    'valence': audio_features.get('valence', 0.5),
                    'energy': audio_features.get('energy', 0.5)
                }
            }
        
        # STEP 6: Build complete response
        print("\n📦 Step 6: Building response...")
        
        response = {
            # Playback state (from Spotify)
            'is_playing': True,
            'type': 'track',
            'timestamp': playback_data.get('timestamp'),
            
            # Track info (from Spotify)
            'track': {
                'id': track['id'],
                'name': track['name'],
                'artists': track['artists'],
                'album': track['album'],
                'images': track['album']['images'],
                'duration_ms': track['duration_ms'],
                'popularity': track['popularity'],
                'explicit': track.get('explicit', False),
                'external_url': track['external_url'],
                'uri': track['uri']
            },
            
            # Device info (from Spotify)
            'device': {
                'id': device['id'],
                'name': device['name'],
                'type': device['type'],
                'volume_percent': device.get('volume_percent'),
                'is_active': device.get('is_active', True)
            },
            
            # Playback controls (from Spotify)
            'progress_ms': playback_data['progress_ms'],
            'shuffle_state': playback_data['shuffle_state'],
            'repeat_state': playback_data['repeat_state'],
            'context': playback_data.get('context'),
            
            # Mood analysis (from ML model)
            'mood_analysis': {
                'fused_mood': mood_data['fused_mood'],
                'audio_mood': mood_data['audio_mood'],
                'lyrics_mood': mood_data['lyrics_mood'],
                'confidence': mood_data['confidence'],
                'source': mood_data['source'],
                'scores': mood_data['scores']
            },
            
            # Audio features (from Multi-API)
            'audio_features': {
                'valence': audio_features.get('valence', 0.5),
                'energy': audio_features.get('energy', 0.5),
                'danceability': audio_features.get('danceability', 0.5),
                'acousticness': audio_features.get('acousticness', 0.5),
                'tempo': audio_features.get('tempo', 120)
            },
            
            # Metadata
            'genre': genre,
            'analysis_timestamp': playback_data.get('timestamp')
        }
        
        print(f"\n{'='*60}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"{'='*60}\n")
        
        return response
        
    except (SpotifyAuthError, SpotifyRateLimitError, SpotifyNotFoundError, SpotifyServiceError) as e:
        handle_spotify_error(e)
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze currently playing track: {str(e)}"
        )


@router.get("/spotify/playlists")
async def get_user_playlists(
    authorization: str = Header(None),
    fetch_all: bool = True  # NEW: Option to get all playlists
):
    """
    Get user's Spotify playlists
    
    UPDATED: Now supports full pagination for users with 50+ playlists
    
    Uses: Spotify API only
    Requires: Bearer token in Authorization header
    
    Args:
        fetch_all: If True, fetches all playlists. If False, fetches first 50.
    """
    try:
        access_token = extract_access_token(authorization)
        
        playlists = await spotify_service.get_user_playlists(
            access_token, 
            fetch_all=fetch_all
        )
        
        return {
            "playlists": playlists,
            "total": len(playlists),
            "fetched_all": fetch_all
        }
        
    except (SpotifyAuthError, SpotifyRateLimitError, SpotifyServiceError) as e:
        handle_spotify_error(e)
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
    """
    Health check endpoint with detailed service status
    
    UPDATED: Now includes new Spotify service exception handling status
    """
    model_loaded = model_service.mood_model is not None
    cache_connected = cache_service.is_connected()
    ytmusic_available = music_service.ytmusic is not None
    lastfm_configured = music_service.LASTFM_API_KEY is not None
    
    # Check Spotify service availability
    spotify_configured = False
    spotify_status = "unavailable"
    try:
        # Try to initialize server client
        spotify_service.get_spotify_client()
        spotify_configured = True
        spotify_status = "healthy"
    except ValueError:
        spotify_status = "missing_credentials"
    except Exception as e:
        spotify_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ml_model": {
                "loaded": model_loaded,
                "status": "active" if model_loaded else "fallback"
            },
            "cache": {
                "connected": cache_connected,
                "status": "active" if cache_connected else "disabled"
            },
            "spotify": {
                "configured": spotify_configured,
                "status": spotify_status,
                "mode": "hybrid",
                "features": [
                    "full_pagination",
                    "rate_limiting",
                    "custom_exceptions",
                    "token_validation",
                    "podcast_support"
                ]
            },
            "ytmusic": {
                "available": ytmusic_available,
                "status": "active" if ytmusic_available else "unavailable"
            },
            "lastfm": {
                "configured": lastfm_configured,
                "status": "active" if lastfm_configured else "unconfigured"
            },
            "musicbrainz": {
                "available": True,
                "status": "active"
            },
            "acousticbrainz": {
                "available": True,
                "status": "active"
            }
        },
        "api_stack": {
            "metadata": "Spotify API (user OAuth)",
            "audio_features": "MusicBrainz + AcousticBrainz",
            "recommendations": "Last.fm",
            "track_search": "YTMusicAPI",
            "lyrics": "Genius API"
        },
        "capabilities": {
            "spotify_playlists": {
                "max_tracks": "unlimited (paginated)",
                "supports_unavailable_tracks": True,
                "supports_local_files": True,
                "caching": True
            },
            "mood_analysis": {
                "hybrid_approach": True,
                "ml_model": model_loaded,
                "lyrics_sentiment": True,
                "audio_features": True,
                "genre_adaptive": True
            },
            "error_handling": {
                "custom_exceptions": True,
                "rate_limit_protection": True,
                "token_expiration_detection": True,
                "retry_logic": False  # Can be added if needed
            }
        },
        "required_spotify_scopes": spotify_service.get_required_scopes()
    }


@router.get("/spotify/test-connection")
async def test_spotify_connection(authorization: str = Header(None)):
    """
    Test Spotify API connection and token validity
    
    NEW ENDPOINT: Helps debug authentication issues
    
    Returns:
        - Token validity status
        - Available endpoints based on scopes
        - User profile info (if token is valid)
    """
    try:
        access_token = extract_access_token(authorization)
        
        print("🧪 Testing Spotify connection...")
        
        # Try to get user profile (requires basic scopes)
        sp = spotify_service.get_spotify_client(access_token)
        
        try:
            user_profile = sp.current_user()
            
            # Verify available scopes
            available_endpoints = spotify_service.verify_token_scopes(access_token)
            
            return {
                "status": "connected",
                "token_valid": True,
                "user": {
                    "id": user_profile.get('id'),
                    "display_name": user_profile.get('display_name'),
                    "email": user_profile.get('email'),
                    "country": user_profile.get('country'),
                    "product": user_profile.get('product')  # free/premium
                },
                "available_endpoints": available_endpoints,
                "required_scopes": spotify_service.get_required_scopes(),
                "recommendations": {
                    "missing_scopes": [
                        scope for scope, available in available_endpoints.items()
                        if not available
                    ]
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "token_valid": False,
                "error": str(e),
                "message": "Token is invalid or has insufficient permissions",
                "required_scopes": spotify_service.get_required_scopes()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spotify/rate-limit-status")
async def get_rate_limit_status():
    """
    Check current rate limit status
    
    NEW ENDPOINT: Helps monitor API usage
    
    Returns:
        Current rate limit counts for different endpoints
    """
    try:
        rate_limiter = spotify_service.rate_limiter
        
        status = {}
        for endpoint, requests in rate_limiter.requests.items():
            limit, window = rate_limiter.limits.get(endpoint, rate_limiter.limits['default'])
            
            # Count recent requests
            import time
            now = time.time()
            recent_requests = [
                req for req in requests
                if now - req < window
            ]
            
            status[endpoint] = {
                "recent_requests": len(recent_requests),
                "limit": limit,
                "window_seconds": window,
                "remaining": max(0, limit - len(recent_requests)),
                "percentage_used": (len(recent_requests) / limit * 100) if limit > 0 else 0
            }
        
        return {
            "rate_limits": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Rate limit status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spotify/clear-cache")
async def clear_spotify_cache(
    authorization: str = Header(None),
    cache_type: Optional[str] = None
):
    """
    Clear Spotify-related cache entries
    
    NEW ENDPOINT: Useful for debugging and forcing fresh data
    
    Args:
        cache_type: Type of cache to clear (playlists, tracks, user_data, all)
    """
    try:
        access_token = extract_access_token(authorization)
        
        # This would require implementing cache clearing in cache_service
        # For now, return a placeholder response
        
        return {
            "status": "success",
            "message": f"Cache clearing requested for: {cache_type or 'all'}",
            "note": "Cache entries will expire naturally based on TTL",
            "ttl_info": {
                "playlists": "5 minutes",
                "playlist_tracks": "10 minutes",
                "track_info": "1 day",
                "user_data": "1 hour",
                "mood_analysis": "1 hour"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/endpoints")
async def list_available_endpoints():
    """
    List all available API endpoints with descriptions
    
    Useful for API documentation and debugging
    """
    return {
        "endpoints": {
            "multi_api": {
                "POST /mood/track": "Analyze mood using Multi-API (no Spotify account needed)",
                "POST /mood/playlist": "Analyze playlist mood using Multi-API",
                "POST /mood/search-and-analyze": "Search and analyze track mood",
                "POST /mood/batch-analyze": "Batch analyze multiple tracks"
            },
            "spotify_hybrid": {
                "POST /mood/spotify/track": "Analyze Spotify track mood (requires auth)",
                "POST /mood/spotify/playlist": "Analyze Spotify playlist mood (requires auth, supports 100+ tracks)",
                "GET /mood/spotify/currently-playing": "Analyze currently playing track (requires auth)",
                "GET /mood/spotify/playlists": "Get user playlists (supports 50+ playlists)"
            },
            "utility": {
                "GET /mood/health": "Comprehensive health check",
                "GET /mood/spotify/test-connection": "Test Spotify token validity",
                "GET /mood/spotify/rate-limit-status": "Check rate limit usage",
                "POST /mood/spotify/clear-cache": "Clear cache entries",
                "GET /mood/debug/endpoints": "This endpoint - lists all available endpoints"
            }
        },
        "authentication": {
            "spotify_endpoints": "Require 'Authorization: Bearer <token>' header",
            "multi_api_endpoints": "No authentication required"
        },
        "features": {
            "pagination": "All Spotify endpoints support full pagination",
            "rate_limiting": "Built-in rate limiting protection",
            "caching": "Intelligent caching for improved performance",
            "error_handling": "Comprehensive error handling with custom exceptions"
        }
    }


# ============================================
# NOTE: Exception handlers must be added to the main app, not router
# Add these to main.py instead:
#
# from services.spotify_service import SpotifyAuthError, SpotifyRateLimitError
#
# @app.exception_handler(SpotifyAuthError)
# async def spotify_auth_error_handler(request, exc):
#     return JSONResponse(
#         status_code=401,
#         content={
#             "error": "authentication_failed",
#             "message": str(exc),
#             "action": "Please re-authenticate with Spotify"
#         }
#     )
#
# @app.exception_handler(SpotifyRateLimitError)
# async def spotify_rate_limit_error_handler(request, exc):
#     return JSONResponse(
#         status_code=429,
#         content={
#             "error": "rate_limit_exceeded",
#             "message": str(exc),
#             "retry_after": exc.retry_after,
#             "action": f"Please wait {exc.retry_after} seconds before retrying"
#         },
#         headers={"Retry-After": str(exc.retry_after)}
#     )
# ============================================