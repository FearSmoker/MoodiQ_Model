"""
Updated Mood Prediction Router - Multi-Mood Support (12 Moods + 2-3 Tags)
- Handles responses with primary_mood and all_moods
- Compatible with 12 extended moods
- Maintains backward compatibility with 4 base moods
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
import time
import traceback

last_log_times = {}

def log_rate_limited(key: str, message: str):
    now = time.time()
    if key not in last_log_times or now - last_log_times[key] > 20:
        last_log_times[key] = now
        print(message)

router = APIRouter()

class TrackMoodRequest(BaseModel):
    
    track_name: str
    artist_name: str
    user_id: Optional[str] = None
    genre: Optional[str] = None

class SpotifyTrackMoodRequest(BaseModel):
    
    track_id: str
    user_id: Optional[str] = None

class TrackMoodResponse(BaseModel):
    
    track_name: str
    artist_name: str
    mood: Dict[str, Any]  # Contains primary_mood, all_moods, mood_scores
    features: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class PlaylistMoodRequest(BaseModel):
    
    tracks: List[Dict[str, str]]
    user_id: Optional[str] = None

class SpotifyPlaylistMoodRequest(BaseModel):
    
    playlist_id: str
    user_id: Optional[str] = None
    include_unavailable: Optional[bool] = False

class PlaylistMoodResponse(BaseModel):
    
    tracks: List[Dict[str, Any]]
    moodDistribution: Dict[str, float]  # Now includes all 12 moods
    overallMood: str

# Helper functions
def extract_access_token(authorization: Optional[str]) -> str:
    
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Expected format: 'Bearer <token>'"
        )
    return authorization.replace('Bearer ', '').strip()

def handle_spotify_error(e: Exception) -> None:
    
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

def format_mood_response(mood_data: Dict) -> Dict:

    primary_mood = mood_data.get('primary_mood') or mood_data.get('fused_mood') or 'Unknown'
    
    # Get all moods (new format)
    all_moods = mood_data.get('all_moods', [primary_mood] if primary_mood != 'Unknown' else [])
    
    # Get mood scores
    mood_scores = mood_data.get('mood_scores', {primary_mood: mood_data.get('confidence')} if primary_mood != 'Unknown' else {})
    
    # Build response
    return {
        'primary_mood': primary_mood,
        'all_moods': all_moods,
        'mood_scores': mood_scores,
        'confidence': mood_data.get('confidence'),
        'base_mood': mood_data.get('base_mood', primary_mood),
        'lyrics_mood': mood_data.get('lyrics_mood'),
        'source': mood_data.get('source', 'ml_model_multi_tag'),
        'scores': mood_data.get('scores', {}),
        'num_tags': len(all_moods),

        'fused_mood': primary_mood,  # Old API compatibility
        'audio_mood': mood_data.get('audio_mood', mood_data.get('base_mood', primary_mood)),
    }

# Original Multi-API endpoints
@router.post("/track", response_model=TrackMoodResponse)
async def get_track_mood(
    request: TrackMoodRequest,
    authorization: str = Header(None)
):
    
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
        
        access_token = None
        if authorization and authorization.startswith('Bearer '):
            access_token = authorization.replace('Bearer ', '')

        # Get audio features
        print("Step 1: Getting audio features...")
        audio_features = await music_service.get_audio_features(
            request.track_name,
            request.artist_name,
            genre=request.genre,
            access_token=access_token
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
        
        # Predict mood (now returns multi-mood tags)
        print("\nStep 4: Predicting mood(s)...")
        mood_data = await model_service.predict_mood_from_features(
            audio_features,
            lyrics_sentiment,
            user_id=request.user_id,
            track_id=None,
            genre=genre
        )
        
        # Format response
        formatted_mood = format_mood_response(mood_data)
        
        print(f"\n✅ FINAL RESULT:")
        print(f"   Primary Mood: {formatted_mood['primary_mood']} ⭐")
        print(f"   All Moods: {', '.join(formatted_mood['all_moods'])}")
        print(f"   Confidence: {formatted_mood['confidence']:.2%}")
        print(f"   Tags: {formatted_mood['num_tags']}")
        print(f"{'='*60}\n")
        
        result = {
            "track_name": request.track_name,
            "artist_name": request.artist_name,
            "mood": formatted_mood,
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
async def get_playlist_mood(
    request: PlaylistMoodRequest,
    authorization: str = Header(None)
):
    
    try:
        print(f"\n{'='*60}")
        print(f"🎵 ANALYZING PLAYLIST: {len(request.tracks)} tracks")
        print(f"{'='*60}\n")
        
        if not request.tracks:
            raise HTTPException(status_code=400, detail="No tracks provided")
            
        access_token = None
        if authorization and authorization.startswith('Bearer '):
            access_token = authorization.replace('Bearer ', '')

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
                    audio_features = await music_service.get_audio_features(
                        track_name, 
                        artist_name,
                        access_token=access_token
                    )
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
                    
                    # Format mood data
                    mood_data = format_mood_response(mood_data)
                    
                    track_cache = {
                        "track_name": track_name,
                        "artist_name": artist_name,
                        "mood": mood_data,
                        "features": audio_features,
                        "tags": tags
                    }
                    await cache_service.set_in_cache(cache_key, track_cache, expiration=3600)
                
                primary_mood = mood_data.get('primary_mood', mood_data.get('fused_mood', 'Relaxed'))
                all_moods = mood_data.get('all_moods', [primary_mood])
                
                print(f"   ✅ Moods: {', '.join(all_moods)}")
                
                processed_tracks.append({
                    "name": track_name,
                    "artist": artist_name,
                    "features": audio_features,
                    "mood": primary_mood,  
                    "primary_mood": primary_mood,
                    "all_moods": all_moods,
                    "mood_scores": mood_data.get('mood_scores', {}),
                    "moodScore": mood_data.get('confidence', 0),
                    "moodDetails": mood_data,
                    "tags": tags[:5]
                })
                
            except Exception as track_error:
                print(f"❌ Error processing track {idx}: {track_error}")
                continue
        
        if not processed_tracks:
            raise HTTPException(status_code=500, detail="Failed to process any tracks")
        
        # Calculate playlist mood distribution (now handles 12 moods)
        mood_stats = model_service.calculate_playlist_mood_distribution(processed_tracks)
        
        print(f"\n✅ PLAYLIST ANALYSIS COMPLETE:")
        print(f"   Processed: {len(processed_tracks)} tracks")
        print(f"   Overall Mood: {mood_stats.get('overall_mood', 'Mixed')}")
        print(f"   Mood Diversity: {mood_stats.get('mood_diversity', 0)} different moods")
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
    
    try:
        access_token = extract_access_token(authorization)
        
        print(f"\n{'='*60}")
        print(f"🎵 ANALYZING SPOTIFY TRACK: {request.track_id}")
        print(f"{'='*60}\n")
        
        # Use the updated predict_mood_from_spotify_track
        mood_data = await model_service.predict_mood_from_spotify_track(
            track_id=request.track_id,
            access_token=access_token,
            lyrics_sentiment={"polarity": 0.0, "subjectivity": 0.0},  # Will be fetched internally
            user_id=request.user_id
        )
        
        # Format response
        formatted_mood = format_mood_response(mood_data)
        
        # Add track info if available
        if 'track_info' in mood_data:
            formatted_mood['track_info'] = mood_data['track_info']
        
        print(f"\n✅ FINAL RESULT:")
        print(f"   Primary Mood: {formatted_mood['primary_mood']} ⭐")
        print(f"   All Moods: {', '.join(formatted_mood['all_moods'])}")
        print(f"   Confidence: {formatted_mood['confidence']:.2%}")
        print(f"{'='*60}\n")
        
        return formatted_mood
        
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
    
    try:
        access_token = extract_access_token(authorization)
        
        log_rate_limited("currently_playing_banner", f"\n{'='*80}\n🎧 ANALYZING CURRENTLY PLAYING TRACK\n{'='*80}\n")
        
        # Get currently playing
        log_rate_limited("currently_playing_step1", "Step 1: Fetching playback state...")
        playback_data = await spotify_service.get_currently_playing(access_token)
        
        if not playback_data or not playback_data.get('is_playing'):
            log_rate_limited("currently_playing_no_track", "❌ No track currently playing")
            return {
                'is_playing': False,
                'message': 'No track currently playing',
                'timestamp': None
            }

        # NOTE: Spotify's `progress_ms` is already the live, up-to-the-moment
        # playback position computed by Spotify at response time. `timestamp`
        # is "when playback state was last changed" (play/pause/seek/skip),
        
        # on top of progress_ms double-counts elapsed time and makes the
        # reported position drift further and further ahead of real playback
        # the longer it's been since the last seek/play event. Trust
        # progress_ms as-is; do not adjust it here.
        
        # Handle podcasts
        if playback_data.get('type') == 'episode':
            log_rate_limited("currently_playing_podcast", "📻 Currently playing: Podcast Episode")
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
        
        log_rate_limited(f"currently_playing_track_{track_id}", f"✅ Playing: {track_name} by {artist_name}\n   Device: {device['name']} ({device['type']})\n   Volume: {device.get('volume_percent')}%")

        # ------------------------------------------------------------------
        # Reuse the cached analysis for this track/user if we already have
        
        # model prediction) ran on EVERY poll (every ~10s), even for the
        # exact same currently-playing song. Any transient variance or
        # partial failure in those 4 external calls produced a *different*
        # mood/confidence/energy/valence each time, which is what caused the
        # "Now Playing" card to flicker between different mood tags, and
        # also seeded the live-history sorted set (used for the Live Mood
        # Graph) with inconsistent values for a single song.
        
        # We still write a fresh history point on every poll below (so the
        # graph keeps getting new timestamped data and stays "live"), but we
        # now source it from the SAME cached mood/features for as long as
        # the same track keeps playing, so the line stays flat and
        # continuous instead of jumping around.
        # ------------------------------------------------------------------
        cache_key = f"track:mood:{track_name}:{artist_name}:{user_id or 'global'}"
        cached_analysis = await cache_service.get_from_cache(cache_key)

        if cached_analysis and cached_analysis.get('mood') and cached_analysis.get('features'):
            log_rate_limited(f"currently_playing_cache_hit_{track_id}", f"📦 Using cached mood analysis for '{track_name}'")
            audio_features = cached_analysis['features']
            formatted_mood = cached_analysis['mood']
            cached_tags = cached_analysis.get('tags') or []
            genre = cached_tags[0] if cached_tags else None
        else:
            # Get audio features
            log_rate_limited("currently_playing_step2", "\nStep 2: Getting audio features...")
            try:
                audio_features = await music_service.get_audio_features(track_name, artist_name, access_token=access_token)
                if not audio_features:
                    audio_features = music_service.get_default_features()

                log_rate_limited(f"currently_playing_features_{track_id}", f"✅ Features: V={audio_features.get('valence', 0):.2f}, E={audio_features.get('energy', 0):.2f}")
            except Exception as e:
                log_rate_limited(f"currently_playing_features_err_{track_id}", f"⚠️ Audio features error: {e}")
                audio_features = music_service.get_default_features()

            # Get lyrics sentiment
            log_rate_limited("currently_playing_step3", "\nStep 3: Analyzing lyrics...")
            try:
                lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
                log_rate_limited(f"currently_playing_lyrics_{track_id}", f"✅ Lyrics: Polarity={lyrics_sentiment.get('polarity', 0):.2f}")
            except Exception as e:
                log_rate_limited(f"currently_playing_lyrics_err_{track_id}", f"⚠️ Lyrics error: {e}")
                lyrics_sentiment = {'polarity': 0.0, 'subjectivity': 0.0}

            # Get genre
            log_rate_limited("currently_playing_step4", "\nStep 4: Getting genre...")
            try:
                tags = await music_service.get_lastfm_tags(track_name, artist_name)
                genre = tags[0] if tags else None
                log_rate_limited(f"currently_playing_genre_{track_id}", f"✅ Genre: {genre or 'Unknown'}")
            except Exception as e:
                log_rate_limited(f"currently_playing_genre_err_{track_id}", f"⚠️ Genre error: {e}")
                genre = None

            # Predict mood (multi-mood)
            log_rate_limited("currently_playing_step5", "\nStep 5: Predicting mood(s)...")
            try:
                mood_data = await model_service.predict_mood_from_features(
                    audio_features,
                    lyrics_sentiment,
                    user_id=user_id,
                    track_id=track_id,
                    genre=genre
                )

                formatted_mood = format_mood_response(mood_data)

                # Cache the successful analysis so subsequent polls for this
                # same track reuse it instead of recomputing (and possibly
                # drifting) each time. Cache under both the name-based key
                # (matches /track and /playlist convention) even for
                # anonymous ('global') users, so the stabilization applies
                # regardless of whether user_id was supplied.
                cache_payload = {
                    "track_name": track_name,
                    "artist_name": artist_name,
                    "mood": formatted_mood,
                    "features": audio_features,
                    "tags": [genre] if genre else []
                }
                await cache_service.set_in_cache(cache_key, cache_payload, expiration=86400)

                log_rate_limited(f"currently_playing_mood_{track_id}", f"\n✅ MOOD PREDICTED:\n   Primary Mood: {formatted_mood['primary_mood']} ⭐\n   All Moods: {', '.join(formatted_mood['all_moods'])}\n   Confidence: {formatted_mood['confidence']:.2%}")

            except Exception as e:
                log_rate_limited(f"currently_playing_mood_err_{track_id}", f"⚠️ Mood prediction error: {e}")
                traceback.print_exc()

                formatted_mood = {
                    'primary_mood': 'Unknown',
                    'all_moods': ['Unknown'],
                    'mood_scores': {'Unknown': 0.0},
                    'confidence': 0.0,
                    'source': 'fallback',
                    'scores': {
                        'valence': audio_features.get('valence', 0.5),
                        'energy': audio_features.get('energy', 0.5)
                    }
                }
                
                # shouldn't get "stuck" and stabilized for the rest of the
                # song. The next poll will retry the full pipeline.

        # Push a fresh, timestamped point to the live-history sorted set on
        # every poll (whether this poll hit cache or computed fresh), using
        # whichever audio_features/formatted_mood we ended up with above.
        # Skip only for genuine fallback failures, so a temporary glitch
        # doesn't inject a bad point into the Live Mood Graph.
        if user_id and formatted_mood.get('source') != 'fallback':
            import time as _time
            history_entry = {
                "track_id": track_id,
                "track_name": track_name,
                "artist_name": artist_name,
                "mood": formatted_mood.get('primary_mood', 'Unknown'),
                "confidence": formatted_mood.get('confidence', 0.0),
                "features": {
                    "valence": audio_features.get('valence', 0.5),
                    "energy": audio_features.get('energy', 0.5),
                    "danceability": audio_features.get('danceability', 0.5),
                    "acousticness": audio_features.get('acousticness', 0.5),
                    "tempo": audio_features.get('tempo', 120),
                }
            }
            await cache_service.zadd_history(user_id, int(_time.time() * 1000), history_entry)
        
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
            
            'mood_analysis': formatted_mood,
            
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
                    track_info = cached_mood.get('track_info', {})
                else:
                    lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(track_name, artist_name)
                    
                    raw_mood_data = await model_service.predict_mood_from_spotify_track(
                        track_id=track_id,
                        access_token=access_token,
                        lyrics_sentiment=lyrics_sentiment,
                        user_id=request.user_id
                    )

                    # IMPORTANT: format_mood_response() only whitelists mood-
                    # classification fields (primary_mood, all_moods, scores,
                    # etc). It does NOT preserve 'track_info' (the audio
                    # feature data — valence/energy/danceability). Grab it
                    # from the raw response before it's discarded, then
                    # stitch it back onto the formatted dict so it survives
                    # both this request and the cache round-trip below.
                    track_info = raw_mood_data.get('track_info', {})
                    mood_data = format_mood_response(raw_mood_data)
                    mood_data['track_info'] = track_info
                    
                    await cache_service.set_in_cache(cache_key, mood_data, expiration=3600)
                
                primary_mood = mood_data.get('primary_mood', 'Relaxed')
                all_moods = mood_data.get('all_moods', [primary_mood])
                
                print(f"   ✅ Moods: {', '.join(all_moods)}")
                
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
                    "mood": primary_mood,  
                    "primary_mood": primary_mood,
                    "all_moods": all_moods,
                    "mood_scores": mood_data.get('mood_scores', {}),
                    "moodScore": mood_data.get('confidence', 0),
                    "moodDetails": mood_data,
                    "features": track_info
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
        
        # Calculate playlist mood distribution (12 moods)
        mood_stats = model_service.calculate_playlist_mood_distribution(processed_tracks)
        
        print(f"\n✅ PLAYLIST ANALYSIS COMPLETE:")
        print(f"   Total: {len(tracks)}")
        print(f"   Processed: {len(processed_tracks)}")
        print(f"   Skipped: {skipped_tracks}")
        print(f"   Overall Mood: {mood_stats.get('overall_mood', 'Mixed')}")
        print(f"   Mood Diversity: {mood_stats.get('mood_diversity', 0)} moods")
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
    
    # Get mood system info
    base_moods = model_service.BASE_MOOD_CLASSES
    extended_moods = model_service.ALL_MOOD_LABELS
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ml_model": {
                "loaded": model_loaded,
                "base_mood_classes": base_moods,
                "extended_mood_classes": extended_moods,
                "total_moods": len(extended_moods),
                "multi_tag_support": True,
                "max_tags_per_track": 3,
                "similarity_threshold": 0.70,
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
                    "podcast_support",
                    "multi_mood_analysis"
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
                "base_moods": len(base_moods),
                "extended_moods": len(extended_moods),
                "multi_tag_classification": True,
                "tags_per_track": "2-3",
                "similarity_threshold": "70%",
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
        "mood_system": {
            "base_moods": base_moods,
            "extended_moods": extended_moods,
            "mapping_approach": "feature_similarity",
            "backward_compatible": True
        },
        "required_spotify_scopes": spotify_service.get_required_scopes()
    }

@router.get("/spotify/test-connection")
async def test_spotify_connection(authorization: str = Header(None)):
    
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
                "mood_system": {
                    "multi_mood_support": True,
                    "extended_moods": model_service.ALL_MOOD_LABELS,
                    "total_moods": len(model_service.ALL_MOOD_LABELS)
                },
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
    cache_type: Optional[str] = None,
    user_id: Optional[str] = None
):
    
    try:
        cleared_keys = []
        errors = []

        # Determine which patterns to delete
        patterns = []
        if cache_type in (None, 'all', 'features'):
            patterns.append('features:*')
        if cache_type in (None, 'all', 'mood'):
            patterns.append('mood:*')
        if cache_type in (None, 'all', 'user_stats') and user_id:
            patterns.append(f'user_stats:{user_id}')
        if cache_type in (None, 'all', 'mbid'):
            patterns.append('mbid:*')
        if cache_type in (None, 'all', 'acousticbrainz'):
            patterns.append('acousticbrainz:*')

        for pattern in patterns:
            try:
                count = await cache_service.delete_by_pattern(pattern)
                cleared_keys.append({'pattern': pattern, 'deleted': count})
            except Exception as pe:
                errors.append({'pattern': pattern, 'error': str(pe)})

        return {
            "status": "success",
            "message": f"Cache cleared for: {cache_type or 'all'}",
            "cleared": cleared_keys,
            "errors": errors
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/endpoints")
async def list_available_endpoints():
    
    return {
        "endpoints": {
            "multi_api": {
                "POST /mood/track": "Analyze mood using Multi-API (no Spotify account needed) - Returns 2-3 mood tags",
                "POST /mood/playlist": "Analyze playlist mood using Multi-API - 12-mood distribution",
                "POST /mood/search-and-analyze": "Search and analyze track mood - Multi-mood support",
                "POST /mood/batch-analyze": "Batch analyze multiple tracks - Multi-mood support"
            },
            "spotify_hybrid": {
                "POST /mood/spotify/track": "Analyze Spotify track mood (requires auth) - Returns 2-3 mood tags",
                "POST /mood/spotify/playlist": "Analyze Spotify playlist mood (requires auth, supports 100+ tracks) - 12-mood distribution",
                "GET /mood/spotify/currently-playing": "Analyze currently playing track (requires auth) - Multi-mood support",
                "GET /mood/spotify/playlists": "Get user playlists (supports 50+ playlists)"
            },
            "utility": {
                "GET /mood/health": "Comprehensive health check - Shows 12-mood system info",
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
            "error_handling": "Comprehensive error handling with custom exceptions",
            "multi_mood_classification": "2-3 mood tags per track from 12 extended moods",
            "similarity_threshold": "70% minimum similarity for mood tags"
        },
        "mood_system": {
            "base_moods": model_service.BASE_MOOD_CLASSES,
            "extended_moods": model_service.ALL_MOOD_LABELS,
            "total_moods": len(model_service.ALL_MOOD_LABELS),
            "tags_per_track": "2-3",
            "similarity_threshold": "70%",
            "mapping_method": "Feature similarity (Euclidean distance)",
            "backward_compatible": True
        }
    }

@router.get("/moods/list")
async def list_available_moods():
    
    try:
        moods_info = {}
        
        for mood_name, mood_data in model_service.EXTENDED_MOODS.items():
            moods_info[mood_name] = {
                "base_moods": mood_data["base_moods"],
                "profile": mood_data["profile"],
                "key_features": list(mood_data["weights"].keys()),
                "description": f"Maps from {', '.join(mood_data['base_moods'])}"
            }
        
        return {
            "base_moods": {
                "list": model_service.BASE_MOOD_CLASSES,
                "count": len(model_service.BASE_MOOD_CLASSES),
                "source": "ML Model (trained)"
            },
            "extended_moods": {
                "list": model_service.ALL_MOOD_LABELS,
                "count": len(model_service.ALL_MOOD_LABELS),
                "source": "Algorithmic Mapping"
            },
            "mood_details": moods_info,
            "classification": {
                "method": "Feature Similarity",
                "tags_per_track": "2-3",
                "min_similarity": "70%",
                "features_used": model_service.MODEL_FEATURE_ORDER
            }
        }
        
    except Exception as e:
        print(f"❌ Error listing moods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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