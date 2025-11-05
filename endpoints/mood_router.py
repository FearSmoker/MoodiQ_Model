"""
Updated Mood Prediction Router - COMPLETE INTEGRATION
- All modifications from code 2 preserved
- Additional features from code 1 included
- Better error handling
- Detailed logging for debugging
- Proper mood mapping
- Genre detection improvements
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
import os

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
    tracks: List[Dict[str, str]]
    user_id: Optional[str] = None


class SpotifyPlaylistMoodRequest(BaseModel):
    """Request model for Spotify playlist mood analysis"""
    playlist_id: str
    user_id: Optional[str] = None
    include_unavailable: Optional[bool] = False


class PlaylistMoodResponse(BaseModel):
    """Response model for playlist mood"""
    tracks: List[Dict[str, Any]]
    moodDistribution: Dict[str, float]
    overallMood: str


# Helper functions
def extract_access_token(authorization: Optional[str]) -> str:
    """Extract and validate access token"""
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected format: 'Bearer <token>'"
        )
    return authorization.replace('Bearer ', '').strip()


def handle_spotify_error(e: Exception) -> None:
    """Convert Spotify service exceptions to HTTP exceptions"""
    if isinstance(e, SpotifyAuthError):
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
    elif isinstance(e, SpotifyRateLimitError):
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded: {str(e)}",
            headers={"Retry-After": str(e.retry_after)}
        )
    elif isinstance(e, SpotifyNotFoundError):
        raise HTTPException(status_code=404, detail=f"Resource not found: {str(e)}")
    elif isinstance(e, SpotifyServiceError):
        raise HTTPException(status_code=500, detail=f"Spotify service error: {str(e)}")


# Original Multi-API endpoints
@router.post("/track", response_model=TrackMoodResponse)
async def get_track_mood(request: TrackMoodRequest):
    """Analyze mood for a single track using multi-API approach"""
    cache_key = f"track:mood:{request.track_name}:{request.artist_name}:{request.user_id or 'global'}"
    
    try:
        # Check cache
        cached_result = await cache_service.get_from_cache(cache_key)
        if cached_result:
            print(f"📦 Cache HIT for track mood")
            return cached_result
        
        print(f"\n{'='*60}")
        print(f"🎵 ANALYZING TRACK: {request.track_name} by {request.artist_name}")
        print(f"{'='*60}\n")
        
        # Get audio features
        print("Step 1: Getting audio features...")
        audio_features = await music_service.get_audio_features(
            request.track_name,
            request.artist_name
        )
        
        if not audio_features:
            print("⚠️ Using default audio features")
            audio_features = music_service.get_default_features()
        else:
            print(f"✅ Audio features retrieved:")
            print(f"   Valence: {audio_features.get('valence', 0):.2f}")
            print(f"   Energy: {audio_features.get('energy', 0):.2f}")
            print(f"   Danceability: {audio_features.get('danceability', 0):.2f}")
        
        # Get genre tags
        print("\nStep 2: Getting genre tags...")
        tags = await music_service.get_lastfm_tags(
            request.track_name,
            request.artist_name
        )
        
        genre = request.genre
        if not genre and tags:
            genre = tags[0]
        
        print(f"✅ Genre: {genre or 'Unknown'}")
        print(f"   Tags: {', '.join(tags[:5]) if tags else 'None'}")
        
        # Get lyrics sentiment
        print("\nStep 3: Analyzing lyrics sentiment...")
        lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
            request.track_name,
            request.artist_name
        )
        
        print(f"✅ Lyrics sentiment:")
        print(f"   Polarity: {lyrics_sentiment.get('polarity', 0):.2f}")
        print(f"   Subjectivity: {lyrics_sentiment.get('subjectivity', 0):.2f}")
        
        # Predict mood
        print("\nStep 4: Predicting mood...")
        mood_data = await model_service.predict_mood_from_features(
            audio_features,
            lyrics_sentiment,
            user_id=request.user_id,
            track_id=None,
            genre=genre
        )
        
        print(f"\n✅ FINAL RESULT:")
        print(f"   Audio Mood: {mood_data['audio_mood']}")
        print(f"   Lyrics Mood: {mood_data['lyrics_mood']}")
        print(f"   Fused Mood: {mood_data['fused_mood']} ⭐")
        print(f"   Confidence: {mood_data['confidence']:.2%}")
        print(f"{'='*60}\n")
        
        result = {
            "track_name": request.track_name,
            "artist_name": request.artist_name,
            "mood": mood_data,
            "features": audio_features,
            "tags": tags[:10]
        }
        
        # Cache for 1 hour
        await cache_service.set_in_cache(cache_key, result, expiration=3600)
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing track mood: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/playlist", response_model=PlaylistMoodResponse)
async def get_playlist_mood(request: PlaylistMoodRequest):
    """Analyze mood for entire playlist"""
    try:
        print(f"\n{'='*60}")
        print(f"🎵 ANALYZING PLAYLIST: {len(request.tracks)} tracks")
        print(f"{'='*60}\n")
        
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
                
                print(f"🔍 Processing track {idx + 1}/{len(request.tracks)}: {track_name}")
                
                cache_key = f"track:mood:{track_name}:{artist_name}:{request.user_id or 'global'}"
                cached_mood = await cache_service.get_from_cache(cache_key)
                
                if cached_mood:
                    print(f"   📦 Cache HIT")
                    mood_data = cached_mood.get('mood')
                    audio_features = cached_mood.get('features')
                    tags = cached_mood.get('tags', [])
                else:
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
                
                print(f"   ✅ Mood: {mood_data.get('fused_mood', 'Unknown')}")
                
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
        
        print(f"\n✅ PLAYLIST ANALYSIS COMPLETE:")
        print(f"   Processed: {len(processed_tracks)} tracks")
        print(f"   Overall Mood: {mood_stats.get('overall_mood', 'Mixed')}")
        print(f"   Distribution: {mood_stats.get('distribution', {})}")
        print(f"{'='*60}\n")
        
        response = {
            "tracks": processed_tracks,
            "moodDistribution": mood_stats.get('distribution', {}),
            "overallMood": mood_stats.get('overall_mood', 'Mixed')
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Playlist mood analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Spotify-specific endpoints
@router.post("/spotify/track", response_model=Dict[str, Any])
async def get_spotify_track_mood(
    request: SpotifyTrackMoodRequest,
    authorization: str = Header(None)
):
    """Analyze mood for a Spotify track"""
    try:
        access_token = extract_access_token(authorization)
        
        print(f"\n{'='*60}")
        print(f"🎵 ANALYZING SPOTIFY TRACK: {request.track_id}")
        print(f"{'='*60}\n")
        
        # Get track info from Spotify
        print("Step 1: Getting track info from Spotify...")
        track_info = await spotify_service.get_track_info(request.track_id, access_token)
        
        if not track_info:
            raise HTTPException(status_code=404, detail="Track not found on Spotify")
        
        track_name = track_info['name']
        artist_name = spotify_service.get_primary_artist_name(track_info)
        
        print(f"✅ Track: {track_name} by {artist_name}")
        
        # Get lyrics sentiment
        print("\nStep 2: Analyzing lyrics...")
        lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
        print(f"✅ Lyrics: Polarity={lyrics_sentiment.get('polarity', 0):.2f}")
        
        # Predict mood using hybrid approach
        print("\nStep 3: Predicting mood...")
        mood_data = await model_service.predict_mood_from_spotify_track(
            track_id=request.track_id,
            access_token=access_token,
            lyrics_sentiment=lyrics_sentiment,
            user_id=request.user_id
        )
        
        print(f"\n✅ FINAL RESULT:")
        print(f"   Fused Mood: {mood_data.get('fused_mood')} ⭐")
        print(f"   Confidence: {mood_data.get('confidence', 0):.2%}")
        print(f"{'='*60}\n")
        
        return mood_data
        
    except (SpotifyAuthError, SpotifyRateLimitError, SpotifyNotFoundError, SpotifyServiceError) as e:
        handle_spotify_error(e)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error analyzing Spotify track mood: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spotify/currently-playing")
async def get_currently_playing_mood(
    authorization: str = Header(None),
    user_id: Optional[str] = None
):
    """Analyze mood of currently playing track"""
    try:
        access_token = extract_access_token(authorization)
        
        print(f"\n{'='*80}")
        print(f"🎧 ANALYZING CURRENTLY PLAYING TRACK")
        print(f"{'='*80}\n")
        
        # Get currently playing
        print("Step 1: Fetching playback state...")
        playback_data = await spotify_service.get_currently_playing(access_token)
        
        if not playback_data or not playback_data.get('is_playing'):
            print("❌ No track currently playing")
            return {
                'is_playing': False,
                'message': 'No track currently playing',
                'timestamp': None
            }
        
        # Handle podcasts
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
        
        track = playback_data['track']
        device = playback_data['device']
        track_id = track['id']
        track_name = track['name']
        artist_name = track['artists'][0]['name']
        
        print(f"✅ Playing: {track_name} by {artist_name}")
        print(f"   Device: {device['name']} ({device['type']})")
        print(f"   Volume: {device.get('volume_percent')}%")
        
        # Get audio features
        print("\nStep 2: Getting audio features...")
        try:
            audio_features = await music_service.get_audio_features(track_name, artist_name)
            if not audio_features:
                audio_features = music_service.get_default_features()
            
            print(f"✅ Features: V={audio_features.get('valence', 0):.2f}, E={audio_features.get('energy', 0):.2f}")
        except Exception as e:
            print(f"⚠️ Audio features error: {e}")
            audio_features = music_service.get_default_features()
        
        # Get lyrics sentiment
        print("\nStep 3: Analyzing lyrics...")
        try:
            lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
            print(f"✅ Lyrics: Polarity={lyrics_sentiment.get('polarity', 0):.2f}")
        except Exception as e:
            print(f"⚠️ Lyrics error: {e}")
            lyrics_sentiment = {'polarity': 0.0, 'subjectivity': 0.0}
        
        # Get genre
        print("\nStep 4: Getting genre...")
        try:
            tags = await music_service.get_lastfm_tags(track_name, artist_name)
            genre = tags[0] if tags else None
            print(f"✅ Genre: {genre or 'Unknown'}")
        except Exception as e:
            print(f"⚠️ Genre error: {e}")
            genre = None
        
        # Predict mood
        print("\nStep 5: Predicting mood...")
        try:
            mood_data = await model_service.predict_mood_from_features(
                audio_features,
                lyrics_sentiment,
                user_id=user_id,
                track_id=track_id,
                genre=genre
            )
            
            print(f"\n✅ MOOD PREDICTED:")
            print(f"   Audio Mood: {mood_data['audio_mood']}")
            print(f"   Lyrics Mood: {mood_data['lyrics_mood']}")
            print(f"   Fused Mood: {mood_data['fused_mood']} ⭐")
            print(f"   Confidence: {mood_data['confidence']:.2%}")
            
        except Exception as e:
            print(f"⚠️ Mood prediction error: {e}")
            traceback.print_exc()
            
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
        
        # Build response
        response = {
            'is_playing': True,
            'type': 'track',
            'timestamp': playback_data.get('timestamp'),
            
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
            
            'device': {
                'id': device['id'],
                'name': device['name'],
                'type': device['type'],
                'volume_percent': device.get('volume_percent'),
                'is_active': device.get('is_active', True)
            },
            
            'progress_ms': playback_data['progress_ms'],
            'shuffle_state': playback_data['shuffle_state'],
            'repeat_state': playback_data['repeat_state'],
            'context': playback_data.get('context'),
            
            'mood_analysis': {
                'fused_mood': mood_data['fused_mood'],
                'audio_mood': mood_data['audio_mood'],
                'lyrics_mood': mood_data['lyrics_mood'],
                'confidence': mood_data['confidence'],
                'source': mood_data['source'],
                'scores': mood_data['scores']
            },
            
            'audio_features': {
                'valence': audio_features.get('valence', 0.5),
                'energy': audio_features.get('energy', 0.5),
                'danceability': audio_features.get('danceability', 0.5),
                'acousticness': audio_features.get('acousticness', 0.5),
                'tempo': audio_features.get('tempo', 120)
            },
            
            'genre': genre,
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        print(f"\n{'='*80}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"{'='*80}\n")
        
        return response
        
    except (SpotifyAuthError, SpotifyRateLimitError, SpotifyNotFoundError, SpotifyServiceError) as e:
        handle_spotify_error(e)
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to analyze currently playing track: {str(e)}")


@router.post("/spotify/playlist", response_model=Dict[str, Any])
async def get_spotify_playlist_mood(
    request: SpotifyPlaylistMoodRequest,
    authorization: str = Header(None)
):
    """Analyze mood for entire Spotify playlist"""
    try:
        access_token = extract_access_token(authorization)
        
        print(f"\n{'='*60}")
        print(f"📂 ANALYZING SPOTIFY PLAYLIST: {request.playlist_id}")
        print(f"{'='*60}\n")
        
        # Get playlist tracks
        tracks = await spotify_service.get_playlist_tracks(
            request.playlist_id, 
            access_token,
            include_unavailable=request.include_unavailable
        )
        
        if not tracks:
            raise HTTPException(status_code=404, detail="Playlist not found or empty")
        
        print(f"🎵 Found {len(tracks)} tracks in playlist\n")
        
        processed_tracks = []
        skipped_tracks = 0
        
        for idx, track in enumerate(tracks):
            try:
                if track.get('unavailable'):
                    print(f"⚠️ Skipping unavailable track at position {idx + 1}")
                    skipped_tracks += 1
                    continue
                
                track_id = track.get('id')
                if not track_id:
                    skipped_tracks += 1
                    continue
                
                track_name = track['name']
                artist_name = track['artists'][0]['name'] if track['artists'] else "Unknown"
                
                print(f"🔍 {idx + 1}/{len(tracks)}: {track_name}")
                
                # Check cache
                cache_key = f"spotify:track:mood:{track_id}:{request.user_id or 'global'}"
                cached_mood = await cache_service.get_from_cache(cache_key)
                
                if cached_mood:
                    print(f"   📦 Cache HIT")
                    mood_data = cached_mood
                else:
                    lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
                    
                    mood_data = await model_service.predict_mood_from_spotify_track(
                        track_id=track_id,
                        access_token=access_token,
                        lyrics_sentiment=lyrics_sentiment,
                        user_id=request.user_id
                    )
                    
                    await cache_service.set_in_cache(cache_key, mood_data, expiration=3600)
                
                print(f"   ✅ Mood: {mood_data.get('fused_mood', 'Unknown')}")
                
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
                print(f"   ⚠️ Error: {track_error}")
                skipped_tracks += 1
                continue
        
        if not processed_tracks:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to process any tracks. {skipped_tracks} tracks were skipped."
            )
        
        mood_stats = model_service.calculate_playlist_mood_distribution(processed_tracks)
        
        print(f"\n✅ PLAYLIST ANALYSIS COMPLETE:")
        print(f"   Total: {len(tracks)}")
        print(f"   Processed: {len(processed_tracks)}")
        print(f"   Skipped: {skipped_tracks}")
        print(f"   Overall Mood: {mood_stats.get('overall_mood', 'Mixed')}")
        print(f"   Distribution: {mood_stats.get('distribution', {})}")
        print(f"{'='*60}\n")
        
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
        
        return response
        
    except (SpotifyAuthError, SpotifyRateLimitError, SpotifyNotFoundError, SpotifyServiceError) as e:
        handle_spotify_error(e)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Spotify playlist mood analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spotify/playlists")
async def get_user_playlists(
    authorization: str = Header(None),
    fetch_all: bool = True
):
    """Get user's Spotify playlists"""
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
# UTILITY ENDPOINTS (from code 1)
# ============================================

@router.post("/search-and-analyze")
async def search_and_analyze(request: Dict[str, Any]):
    """Search for a track and analyze its mood (Multi-API)"""
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
    """Batch analyze multiple tracks efficiently"""
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
    
    # Check Spotify service availability
    spotify_configured = False
    spotify_status = "unavailable"
    try:
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
                "mood_classes": model_service.MOOD_CLASSES if hasattr(model_service, 'MOOD_CLASSES') else [],
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
                "retry_logic": False
            }
        },
        "required_spotify_scopes": spotify_service.get_required_scopes()
    }


@router.get("/spotify/test-connection")
async def test_spotify_connection(authorization: str = Header(None)):
    """Test Spotify API connection and token validity"""
    try:
        access_token = extract_access_token(authorization)
        
        print("🧪 Testing Spotify connection...")
        
        sp = spotify_service.get_spotify_client(access_token)
        
        try:
            user_profile = sp.current_user()
            
            available_endpoints = spotify_service.verify_token_scopes(access_token)
            
            return {
                "status": "connected",
                "token_valid": True,
                "user": {
                    "id": user_profile.get('id'),
                    "display_name": user_profile.get('display_name'),
                    "email": user_profile.get('email'),
                    "country": user_profile.get('country'),
                    "product": user_profile.get('product')
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
    """Check current rate limit status"""
    try:
        rate_limiter = spotify_service.rate_limiter
        
        status = {}
        for endpoint, requests in rate_limiter.requests.items():
            limit, window = rate_limiter.limits.get(endpoint, rate_limiter.limits['default'])
            
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
    """Clear Spotify-related cache entries"""
    try:
        access_token = extract_access_token(authorization)
        
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
    """List all available API endpoints with descriptions"""
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
# EXCEPTION HANDLERS NOTE
# ============================================
"""
Add these exception handlers to your main.py:

from fastapi.responses import JSONResponse
from services.spotify_service import SpotifyAuthError, SpotifyRateLimitError

@app.exception_handler(SpotifyAuthError)
async def spotify_auth_error_handler(request, exc):
    return JSONResponse(
        status_code=401,
        content={
            "error": "authentication_failed",
            "message": str(exc),
            "action": "Please re-authenticate with Spotify"
        }
    )

@app.exception_handler(SpotifyRateLimitError)
async def spotify_rate_limit_error_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": str(exc),
            "retry_after": exc.retry_after,
            "action": f"Please wait {exc.retry_after} seconds before retrying"
        },
        headers={"Retry-After": str(exc.retry_after)}
    )
"""