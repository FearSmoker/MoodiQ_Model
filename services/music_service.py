"""
Multi-API Music Service for MoodiQ-AI
Integrates: YTMusicAPI, Last.fm, AcousticBrainz, MusicBrainz, Gemini AI

UPDATED: Fixed Last.fm initialization to load after .env
"""

import os
import requests
import musicbrainzngs
from ytmusicapi import YTMusic
from typing import List, Dict, Optional, Any
from . import cache_service
from . import gemini_service

# ============================================
# API Configuration - Delayed Loading
# ============================================
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

# Initialize these at module level
LASTFM_API_KEY = None
LASTFM_API_SECRET = None
_lastfm_initialized = False

def _init_lastfm():
    """Initialize Last.fm API (called lazily on first use)"""
    global LASTFM_API_KEY, LASTFM_API_SECRET, _lastfm_initialized
    
    if _lastfm_initialized:
        return
    
    _lastfm_initialized = True
    LASTFM_API_KEY = os.getenv("LASTFM_API_KEY") or os.getenv("API_KEY_LASTFM")
    LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET") or os.getenv("API_SECRET_LASTFM")
    
    if LASTFM_API_KEY:
        print(f"✅ Last.fm API configured")
    else:
        print("⚠️  Last.fm API key not found - recommendations will be limited")
# MusicBrainz Configuration
musicbrainzngs.set_useragent(
    "MoodiQ",
    "1.0",
    "https://moodiq.netlify.app"
)

# AcousticBrainz Base URL
ACOUSTICBRAINZ_BASE_URL = "https://acousticbrainz.org/api/v1"

# Initialize YTMusic (no auth needed for search)
ytmusic = None

def init_ytmusic():
    """Initialize YTMusic client"""
    global ytmusic
    try:
        ytmusic = YTMusic()
        print("✅ YTMusic initialized")
    except Exception as e:
        print(f"⚠️ YTMusic initialization failed: {e}")
        ytmusic = None

# Initialize on import
init_ytmusic()


# ============================================
# 1. Track Search (YTMusicAPI)
# ============================================

async def search_tracks(
    query: str,
    limit: int = 20
) -> List[Dict]:
    """
    Search for tracks using YTMusicAPI
    
    Args:
        query: Search query
        limit: Maximum results
        
    Returns:
        List of track dictionaries
    """
    cache_key = f"ytmusic:search:{query}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: YTMusic search")
        return cached
    
    if not ytmusic:
        print("⚠️ YTMusic not initialized")
        return []
    
    try:
        print(f"🔍 Searching YTMusic: {query}")
        results = ytmusic.search(query, filter="songs", limit=limit)
        
        tracks = []
        for item in results:
            track = {
                'id': item.get('videoId'),
                'name': item.get('title'),
                'artists': [artist['name'] for artist in item.get('artists', [])],
                'artist_ids': [artist.get('id') for artist in item.get('artists', [])],
                'album': item.get('album', {}).get('name') if item.get('album') else None,
                'duration_ms': item.get('duration_seconds', 0) * 1000,
                'thumbnail': item.get('thumbnails', [{}])[0].get('url'),
                'source': 'youtube_music'
            }
            tracks.append(track)
        
        # Cache for 1 hour
        await cache_service.set_in_cache(cache_key, tracks, expiration=3600)
        
        print(f"✅ Found {len(tracks)} tracks on YTMusic")
        return tracks
        
    except Exception as e:
        print(f"❌ YTMusic search error: {e}")
        return []


async def get_track_info(
    track_id: str = None,
    track_name: str = None,
    artist_name: str = None
) -> Optional[Dict]:
    """
    Get track information from YTMusicAPI
    
    Args:
        track_id: YouTube video ID
        track_name: Track name (alternative lookup)
        artist_name: Artist name (for alternative lookup)
        
    Returns:
        Track information dictionary
    """
    if track_id:
        cache_key = f"ytmusic:track:{track_id}"
    else:
        cache_key = f"ytmusic:track:{track_name}:{artist_name}"
    
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached
    
    try:
        if track_id and ytmusic:
            # Get by ID
            song = ytmusic.get_song(track_id)
            
            info = {
                'id': song.get('videoId'),
                'name': song.get('title'),
                'artists': [artist['name'] for artist in song.get('artists', [])],
                'album': song.get('album', {}).get('name'),
                'duration_ms': song.get('lengthSeconds', 0) * 1000,
                'thumbnail': song.get('thumbnails', [{}])[-1].get('url'),
                'source': 'youtube_music'
            }
            
        elif track_name and artist_name:
            # Search and get first result
            query = f"{track_name} {artist_name}"
            results = await search_tracks(query, limit=1)
            
            if not results:
                return None
            
            info = results[0]
        else:
            return None
        
        # Cache for 1 day
        await cache_service.set_in_cache(cache_key, info, expiration=86400)
        return info
        
    except Exception as e:
        print(f"❌ Error getting track info: {e}")
        return None

# ============================================
# 2. MusicBrainz Integration (Get MBID)
# ============================================

async def get_musicbrainz_id(
    track_name: str,
    artist_name: str
) -> Optional[str]:
    """
    Get MusicBrainz ID (MBID) for a track
    
    Args:
        track_name: Track name
        artist_name: Artist name
        
    Returns:
        MBID string or None
    """
    cache_key = f"mbid:{track_name}:{artist_name}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: MBID")
        return cached
    
    try:
        print(f"🔍 Searching MusicBrainz: {track_name} by {artist_name}")
        
        result = musicbrainzngs.search_recordings(
            recording=track_name,
            artist=artist_name,
            limit=1
        )
        
        if result['recording-list']:
            mbid = result['recording-list'][0]['id']
            print(f"✅ Found MBID: {mbid}")
            
            # Cache for 1 week (MBIDs don't change)
            await cache_service.set_in_cache(cache_key, mbid, expiration=604800)
            return mbid
        
        print(f"⚠️ No MBID found for {track_name}")
        return None
        
    except Exception as e:
        print(f"❌ MusicBrainz error: {e}")
        return None


# ============================================
# 3. AcousticBrainz Integration (Audio Features)
# ============================================

async def get_audio_features_from_mbid(mbid: str) -> Optional[Dict]:
    """
    Get audio features from AcousticBrainz using MBID
    
    Args:
        mbid: MusicBrainz ID
        
    Returns:
        Audio features dictionary (normalized to Spotify-like format)
    """
    cache_key = f"acousticbrainz:features:{mbid}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: AcousticBrainz features")
        return cached
    
    try:
        print(f"🎹 Fetching AcousticBrainz features for MBID: {mbid}")
        
        # Get low-level features
        url = f"{ACOUSTICBRAINZ_BASE_URL}/{mbid}/low-level"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            print(f"⚠️ No AcousticBrainz data for MBID: {mbid}")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Extract and normalize features
        features = normalize_acousticbrainz_features(data)
        
        # Cache for 1 week
        await cache_service.set_in_cache(cache_key, features, expiration=604800)
        
        print(f"✅ Retrieved AcousticBrainz features")
        return features
        
    except requests.exceptions.Timeout:
        print(f"⏰ AcousticBrainz timeout for MBID: {mbid}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ AcousticBrainz request error: {e}")
        return None
    except Exception as e:
        print(f"❌ AcousticBrainz error: {e}")
        return None


def normalize_acousticbrainz_features(data: Dict) -> Dict:
    """
    Normalize AcousticBrainz features to Spotify-like format
    FIX: Better error handling and safer dictionary access
    """
    try:
        # Helper function for safe nested dict access
        def safe_get(nested_dict, *keys, default=0.5):
            """Safely navigate nested dictionaries"""
            try:
                result = nested_dict
                for key in keys:
                    if isinstance(result, dict):
                        result = result.get(key, {})
                    else:
                        return default
                
                # Handle the final value
                if isinstance(result, dict):
                    # If it's still a dict, try to get 'mean' or 'value'
                    if 'mean' in result:
                        return float(result['mean'])
                    elif 'value' in result:
                        return float(result['value'])
                    else:
                        return default
                else:
                    return float(result) if result is not None else default
            except (KeyError, TypeError, ValueError):
                return default
        
        # Extract with safe access
        features = {
            'id': safe_get(data, 'metadata', 'tags', 'musicbrainz_recordingid', default=''),
            
            # Valence: Use mood_happy if available
            'valence': safe_get(data, 'highlevel', 'mood_happy', 'probability', default=0.5),
            
            # Energy: Use dynamic_complexity
            'energy': min(1.0, safe_get(data, 'lowlevel', 'dynamic_complexity', default=0.5)),
            
            # Danceability: Direct mapping
            'danceability': safe_get(data, 'rhythm', 'danceability', default=0.5),
            
            # Acousticness: Inverse of spectral brightness
            'acousticness': 1.0 - min(1.0, safe_get(data, 'lowlevel', 'spectral_centroid', default=2000) / 4000),
            
            # Instrumentalness: Use voice/instrumental classifier
            'instrumentalness': 1.0 - safe_get(data, 'highlevel', 'voice_instrumental', 'probability', default=0.5),
            
            # Speechiness: Estimate from spectral features
            'speechiness': min(1.0, safe_get(data, 'lowlevel', 'spectral_rolloff', default=2000) / 8000),
            
            # Tempo (BPM)
            'tempo': safe_get(data, 'rhythm', 'bpm', default=120.0),
            
            # Loudness (dB)
            'loudness': safe_get(data, 'lowlevel', 'loudness', default=-10.0),
            
            # Liveness: Estimate from noise
            'liveness': min(1.0, safe_get(data, 'lowlevel', 'average_loudness', default=0.2)),
            
            # Key: From tonal features
            'key': int(safe_get(data, 'tonal', 'key_key', default=0)) % 12,
            
            # Mode: Major (1) or Minor (0) - safe handling
            'mode': 1,  # Default to major
            
            # Time signature: Default to 4/4
            'time_signature': 4,
            
            # Duration
            'duration_ms': int(safe_get(data, 'metadata', 'audio_properties', 'length', default=0) * 1000),
            
            # Add source tag
            'source': 'acousticbrainz'
        }
        
        # Ensure all values are in valid ranges
        for key in ['valence', 'energy', 'danceability', 'acousticness', 
                    'instrumentalness', 'speechiness', 'liveness']:
            features[key] = max(0.0, min(1.0, features[key]))
        
        features['tempo'] = max(0.0, min(300.0, features['tempo']))
        features['loudness'] = max(-60.0, min(0.0, features['loudness']))
        features['key'] = max(0, min(11, features['key']))
        
        return features
        
    except Exception as e:
        print(f"❌ Feature normalization error: {e}")
        # Return default features on any error
        return get_default_features()


def get_default_features() -> Dict:
    """Return default audio features when data unavailable"""
    return {
        'id': None,
        'valence': 0.5,
        'energy': 0.5,
        'danceability': 0.5,
        'acousticness': 0.5,
        'instrumentalness': 0.3,
        'speechiness': 0.1,
        'tempo': 120.0,
        'loudness': -10.0,
        'liveness': 0.1,
        'key': 0,
        'mode': 1,
        'time_signature': 4,
        'duration_ms': 0,
        'source': 'default'
    }


async def get_audio_features(
    track_name: str,
    artist_name: str,
    genre: Optional[str] = None,
    use_gemini_fallback: bool = True
) -> Optional[Dict]:
    """
    Complete pipeline: Get audio features via MusicBrainz → AcousticBrainz → Gemini AI
    
    🆕 UPDATED: Now includes Gemini AI fallback when MBID not found
    
    Args:
        track_name: Track name
        artist_name: Artist name
        genre: Genre hint (helps Gemini estimation)
        use_gemini_fallback: Whether to use Gemini when MBID/AcousticBrainz fails
        
    Returns:
        Audio features dictionary
    """
    # Check combined cache first
    cache_key = f"features:{track_name}:{artist_name}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: Audio features (source: {cached.get('source', 'unknown')})")
        return cached
    
    # Step 1: Try to get MBID from MusicBrainz
    mbid = await get_musicbrainz_id(track_name, artist_name)
    
    if mbid:
        # Step 2: Try to get features from AcousticBrainz
        features = await get_audio_features_from_mbid(mbid)
        
        if features:
            print(f"✅ Got features from AcousticBrainz")
            # Cache for 1 week (reliable data)
            await cache_service.set_in_cache(cache_key, features, expiration=604800)
            return features
        else:
            print(f"⚠️ No AcousticBrainz data for MBID: {mbid}")
    else:
        print(f"⚠️ No MBID found for {track_name} by {artist_name}")
    
    # Step 3: 🤖 FALLBACK TO GEMINI AI (NEW!)
    if use_gemini_fallback:
        print(f"🤖 Trying Gemini AI fallback...")
        
        # Get genre if not provided
        if not genre:
            tags = await get_lastfm_tags(track_name, artist_name)
            genre = tags[0] if tags else None
        
        gemini_features = await gemini_service.estimate_audio_features_with_gemini(
            track_name,
            artist_name,
            genre=genre
        )
        
        if gemini_features:
            print(f"✅ Got estimated features from Gemini AI")
            # Cache for 6 hours (AI-estimated, less reliable than actual audio analysis)
            await cache_service.set_in_cache(cache_key, gemini_features, expiration=21600)
            return gemini_features
        else:
            print(f"⚠️ Gemini estimation also failed")
    
    # Step 4: Final fallback to defaults
    print(f"⚠️ Using default features (no data sources available)")
    features = get_default_features()
    
    # Cache defaults for shorter time (1 hour) - allows retry sooner
    await cache_service.set_in_cache(cache_key, features, expiration=3600)
    
    return features


async def get_audio_features_enhanced(
    track_name: str,
    artist_name: str,
    album_name: Optional[str] = None,
    lyrics_snippet: Optional[str] = None
) -> Optional[Dict]:
    """
    🆕 Enhanced audio feature extraction with maximum context for Gemini
    
    Use this when you have additional metadata (album, lyrics) for better AI estimation
    
    Args:
        track_name: Track name
        artist_name: Artist name
        album_name: Album name (helps Gemini)
        lyrics_snippet: First few lines of lyrics (helps Gemini significantly)
        
    Returns:
        Audio features dictionary
    """
    # Try standard pipeline first
    features = await get_audio_features(track_name, artist_name, use_gemini_fallback=False)
    
    # If we got real data (not defaults), return it
    if features and features.get('source') in ['acousticbrainz', 'spotify']:
        return features
    
    # Otherwise, use enhanced Gemini estimation with all context
    print(f"🤖 Using enhanced Gemini estimation with full context...")
    
    # Get genre tags
    tags = await get_lastfm_tags(track_name, artist_name)
    genre = tags[0] if tags else None
    
    gemini_features = await gemini_service.estimate_audio_features_with_gemini(
        track_name,
        artist_name,
        album_name=album_name,
        genre=genre,
        lyrics_snippet=lyrics_snippet
    )
    
    if gemini_features:
        print(f"✅ Got enhanced Gemini features")
        cache_key = f"features:{track_name}:{artist_name}"
        await cache_service.set_in_cache(cache_key, gemini_features, expiration=21600)
        return gemini_features
    
    # Final fallback
    return get_default_features()


# ============================================
# 4. Last.fm Integration (Tags, Recommendations, Similar Artists)
# ============================================

async def get_lastfm_tags(
    track_name: str,
    artist_name: str
) -> List[str]:
    """
    Get genre/mood tags from Last.fm
    
    Args:
        track_name: Track name
        artist_name: Artist name
        
    Returns:
        List of tags
    """
    _init_lastfm() 
    cache_key = f"lastfm:tags:{track_name}:{artist_name}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    if not LASTFM_API_KEY:
        print("⚠️ Last.fm API key not configured")
        return []
    
    try:
        print(f"🏷️ Fetching Last.fm tags: {track_name} by {artist_name}")
        
        params = {
            'method': 'track.getTopTags',
            'artist': artist_name,
            'track': track_name,
            'api_key': LASTFM_API_KEY,
            'format': 'json'
        }
        
        response = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        tags = []
        if 'toptags' in data and 'tag' in data['toptags']:
            tags = [tag['name'].lower() for tag in data['toptags']['tag'][:10]]
        
        print(f"✅ Found {len(tags)} tags")
        
        # Cache for 1 week
        await cache_service.set_in_cache(cache_key, tags, expiration=604800)
        return tags
        
    except Exception as e:
        print(f"❌ Last.fm tags error: {e}")
        return []


async def get_similar_tracks_lastfm(
    track_name: str,
    artist_name: str,
    limit: int = 20
) -> List[Dict]:
    """
    Get similar tracks from Last.fm (REPLACEMENT for Spotify Recommendations)
    
    Args:
        track_name: Track name
        artist_name: Artist name
        limit: Maximum results
        
    Returns:
        List of similar tracks
    """
    _init_lastfm() 
    cache_key = f"lastfm:similar:{track_name}:{artist_name}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: Last.fm similar tracks")
        return cached
    
    if not LASTFM_API_KEY:
        print("⚠️ Last.fm API key not configured")
        return []
    
    try:
        print(f"🎯 Fetching similar tracks from Last.fm")
        
        params = {
            'method': 'track.getSimilar',
            'artist': artist_name,
            'track': track_name,
            'api_key': LASTFM_API_KEY,
            'limit': limit,
            'format': 'json'
        }
        
        response = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        similar_tracks = []
        if 'similartracks' in data and 'track' in data['similartracks']:
            for track in data['similartracks']['track']:
                similar_tracks.append({
                    'name': track.get('name'),
                    'artist': track.get('artist', {}).get('name'),
                    'match_score': float(track.get('match', 0)),
                    'url': track.get('url'),
                    'source': 'lastfm'
                })
        
        print(f"✅ Found {len(similar_tracks)} similar tracks")
        
        # Cache for 1 day
        await cache_service.set_in_cache(cache_key, similar_tracks, expiration=86400)
        return similar_tracks
        
    except Exception as e:
        print(f"❌ Last.fm similar tracks error: {e}")
        return []


async def get_similar_artists_lastfm(
    artist_name: str,
    limit: int = 10
) -> List[Dict]:
    """
    Get similar artists from Last.fm (REPLACEMENT for Spotify Related Artists)
    
    Args:
        artist_name: Artist name
        limit: Maximum results
        
    Returns:
        List of similar artists
    """
    _init_lastfm() 
    cache_key = f"lastfm:similar_artists:{artist_name}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    if not LASTFM_API_KEY:
        print("⚠️ Last.fm API key not configured")
        return []
    
    try:
        print(f"🎸 Fetching similar artists from Last.fm")
        
        params = {
            'method': 'artist.getSimilar',
            'artist': artist_name,
            'api_key': LASTFM_API_KEY,
            'limit': limit,
            'format': 'json'
        }
        
        response = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        similar_artists = []
        if 'similarartists' in data and 'artist' in data['similarartists']:
            for artist in data['similarartists']['artist']:
                similar_artists.append({
                    'name': artist.get('name'),
                    'match_score': float(artist.get('match', 0)),
                    'url': artist.get('url'),
                    'source': 'lastfm'
                })
        
        print(f"✅ Found {len(similar_artists)} similar artists")
        
        # Cache for 1 week
        await cache_service.set_in_cache(cache_key, similar_artists, expiration=604800)
        return similar_artists
        
    except Exception as e:
        print(f"❌ Last.fm similar artists error: {e}")
        return []


async def get_recommendations(
    seed_track_name: str,
    seed_artist_name: str,
    target_mood: Optional[str] = None,
    limit: int = 20
) -> List[Dict]:
    """
    Get recommendations combining Last.fm similar tracks + mood filtering
    REPLACEMENT for Spotify Recommendations endpoint
    
    Args:
        seed_track_name: Seed track name
        seed_artist_name: Seed artist name
        target_mood: Target mood (Happy, Sad, Calm, Energetic)
        limit: Maximum results
        
    Returns:
        List of recommended tracks with features
    """
    print(f"🎯 Getting recommendations for {seed_track_name} by {seed_artist_name}")
    
    # Get similar tracks from Last.fm
    similar_tracks = await get_similar_tracks_lastfm(
        seed_track_name,
        seed_artist_name,
        limit=limit * 2  # Get more for filtering
    )
    
    if not similar_tracks:
        print("⚠️ No similar tracks found")
        return []
    
    # Enrich with audio features
    recommendations = []
    for track in similar_tracks[:limit]:
        # Get audio features (will use Gemini if needed!)
        features = await get_audio_features(
            track['name'],
            track['artist']
        )
        
        if features:
            track['features'] = features
            
            # Filter by mood if specified
            if target_mood:
                # Simple mood filtering based on valence/energy
                mood_match = check_mood_match(features, target_mood)
                if not mood_match:
                    continue
            
            recommendations.append(track)
    
    print(f"✅ Generated {len(recommendations)} recommendations")
    return recommendations


def check_mood_match(features: Dict, target_mood: str) -> bool:
    """Check if features match target mood"""
    valence = features.get('valence', 0.5)
    energy = features.get('energy', 0.5)
    
    if target_mood == "Happy":
        return valence > 0.6 and energy > 0.5
    elif target_mood == "Sad":
        return valence < 0.4 and energy < 0.5
    elif target_mood == "Energetic":
        return energy > 0.7
    elif target_mood == "Calm":
        return energy < 0.4
    else:
        return True  # No filtering


# ============================================
# 5. Batch Operations
# ============================================

async def batch_get_audio_features(
    tracks: List[Dict]
) -> List[Optional[Dict]]:
    """
    Get audio features for multiple tracks
    🆕 Now uses Gemini fallback automatically
    
    Args:
        tracks: List of dicts with 'name' and 'artist' keys
        
    Returns:
        List of feature dictionaries
    """
    import asyncio
    
    print(f"🎹 Batch fetching features for {len(tracks)} tracks")
    
    async def fetch_single(track):
        return await get_audio_features(
            track.get('name'),
            track.get('artist')
        )
    
    # Limit concurrent requests
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_with_limit(track):
        async with semaphore:
            return await fetch_single(track)
    
    tasks = [fetch_with_limit(track) for track in tracks]
    features_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    results = []
    for i, result in enumerate(features_list):
        if isinstance(result, Exception):
            print(f"⚠️ Error fetching features for track {i}: {result}")
            results.append(get_default_features())
        else:
            results.append(result)
    
    # Count sources
    sources = {}
    for r in results:
        source = r.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"✅ Batch fetch complete:")
    for source, count in sources.items():
        print(f"   - {source}: {count} tracks")
    
    return results


# ============================================
# Testing Function
# ============================================

async def test_music_service():
    """Test all music service functions"""
    print("\n" + "="*60)
    print("🧪 Testing Multi-API Music Service (with Gemini)")
    print("="*60)
    
    test_track = "Happy"
    test_artist = "Pharrell Williams"
    
    print(f"\n1. Testing YTMusic search...")
    tracks = await search_tracks(f"{test_track} {test_artist}", limit=3)
    print(f"   Found {len(tracks)} tracks")
    
    print(f"\n2. Testing MusicBrainz MBID lookup...")
    mbid = await get_musicbrainz_id(test_track, test_artist)
    print(f"   MBID: {mbid}")
    
    print(f"\n3. Testing audio features (with Gemini fallback)...")
    features = await get_audio_features(test_track, test_artist)
    if features:
        print(f"   Source: {features.get('source', 'unknown')}")
        print(f"   Valence: {features['valence']:.3f}")
        print(f"   Energy: {features['energy']:.3f}")
        print(f"   Tempo: {features['tempo']:.1f} BPM")
    
    print(f"\n4. Testing with obscure song (should trigger Gemini)...")
    obscure_features = await get_audio_features("Some Random Song XYZ", "Unknown Artist ABC")
    if obscure_features:
        print(f"   Source: {obscure_features.get('source', 'unknown')}")
        print(f"   Valence: {obscure_features['valence']:.3f}")
    
    print(f"\n5. Testing Last.fm tags...")
    tags = await get_lastfm_tags(test_track, test_artist)
    print(f"   Tags: {', '.join(tags[:5])}")
    
    print(f"\n6. Testing Last.fm similar tracks...")
    similar = await get_similar_tracks_lastfm(test_track, test_artist, limit=5)
    print(f"   Similar tracks: {len(similar)}")
    for track in similar[:3]:
        print(f"   - {track['name']} by {track['artist']} (match: {track['match_score']:.2f})")
    
    print(f"\n7. Testing recommendations...")
    recommendations = await get_recommendations(test_track, test_artist, limit=5)
    print(f"   Recommendations: {len(recommendations)}")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_music_service())