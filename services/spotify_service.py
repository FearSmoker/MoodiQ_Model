"""
Hybrid Spotify Service
Uses Spotify API for: metadata, playlists, user data, currently playing
Uses Multi-API for: audio features, recommendations, mood analysis
"""

import os
from typing import List, Dict, Optional, Any
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from . import cache_service


# Global Spotify client
sp_server: Optional[spotipy.Spotify] = None


def get_spotify_client(access_token: Optional[str] = None) -> spotipy.Spotify:
    """
    Get Spotify client with user token or server credentials
    
    Args:
        access_token: User's access token (from OAuth)
        
    Returns:
        Configured Spotify client
    """
    if access_token:
        # Use user's access token (from frontend OAuth)
        return spotipy.Spotify(auth=access_token)
    
    # Use server credentials for public endpoints
    global sp_server
    
    if sp_server is None:
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set"
            )
        
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            
            sp_server = spotipy.Spotify(auth_manager=auth_manager)
            print("✅ Spotify server client initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize Spotify client: {e}")
            raise
    
    return sp_server


# ============================================
# USER DATA & PLAYLISTS (Spotify API - Working)
# ============================================

async def get_user_playlists(
    access_token: str,
    limit: int = 50
) -> List[Dict]:
    """
    Get user's playlists from Spotify
    
    Endpoint: GET /me/playlists (WORKING - not restricted)
    """
    cache_key = f"spotify:playlists:{access_token[:10]}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: User playlists")
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        
        print(f"📂 Fetching user playlists from Spotify...")
        playlists_data = sp.current_user_playlists(limit=limit)
        
        playlists = []
        for item in playlists_data['items']:
            playlists.append({
                'id': item['id'],
                'name': item['name'],
                'description': item.get('description'),
                'images': item.get('images', []),
                'tracks_total': item['tracks']['total'],
                'owner': item['owner']['display_name'],
                'public': item.get('public', False),
                'collaborative': item.get('collaborative', False),
                'external_url': item['external_urls']['spotify'],
                'uri': item['uri']
            })
        
        # Cache for 5 minutes
        await cache_service.set_in_cache(cache_key, playlists, expiration=300)
        
        print(f"✅ Retrieved {len(playlists)} playlists from Spotify")
        return playlists
        
    except Exception as e:
        print(f"❌ Error fetching playlists: {e}")
        return []


async def get_playlist_tracks(
    playlist_id: str,
    access_token: str
) -> List[Dict]:
    """
    Get tracks from a Spotify playlist
    
    Endpoint: GET /playlists/{id}/tracks (WORKING - not restricted)
    """
    cache_key = f"spotify:playlist_tracks:{playlist_id}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: Playlist tracks")
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        
        print(f"🎵 Fetching playlist tracks from Spotify...")
        results = sp.playlist_tracks(playlist_id)
        
        tracks = []
        for item in results['items']:
            if not item['track']:
                continue
            
            track = item['track']
            tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name']
                    } for artist in track['artists']
                ],
                'album': {
                    'id': track['album']['id'],
                    'name': track['album']['name'],
                    'images': track['album']['images']
                },
                'duration_ms': track['duration_ms'],
                'popularity': track.get('popularity', 0),
                'explicit': track.get('explicit', False),
                'preview_url': track.get('preview_url'),
                'external_url': track['external_urls']['spotify'],
                'uri': track['uri'],
                'added_at': item.get('added_at')
            })
        
        # Cache for 10 minutes
        await cache_service.set_in_cache(cache_key, tracks, expiration=600)
        
        print(f"✅ Retrieved {len(tracks)} tracks from Spotify")
        return tracks
        
    except Exception as e:
        print(f"❌ Error fetching playlist tracks: {e}")
        return []


async def get_track_info(
    track_id: str,
    access_token: Optional[str] = None
) -> Optional[Dict]:
    """
    Get track information from Spotify
    
    Endpoint: GET /tracks/{id} (WORKING - not restricted)
    """
    cache_key = f"spotify:track:{track_id}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        track = sp.track(track_id)
        
        if not track:
            return None
        
        info = {
            'id': track['id'],
            'name': track['name'],
            'artists': [
                {
                    'id': artist['id'],
                    'name': artist['name']
                } for artist in track['artists']
            ],
            'album': {
                'id': track['album']['id'],
                'name': track['album']['name'],
                'images': track['album']['images'],
                'release_date': track['album'].get('release_date')
            },
            'duration_ms': track['duration_ms'],
            'popularity': track.get('popularity', 0),
            'explicit': track.get('explicit', False),
            'preview_url': track.get('preview_url'),
            'external_url': track['external_urls']['spotify'],
            'uri': track['uri'],
            'disc_number': track.get('disc_number', 1),
            'track_number': track.get('track_number', 1)
        }
        
        # Cache for 1 day
        await cache_service.set_in_cache(cache_key, info, expiration=86400)
        return info
        
    except Exception as e:
        print(f"❌ Error getting track info: {e}")
        return None


async def get_currently_playing(access_token: str) -> Optional[Dict]:
    """
    Get currently playing track
    
    Endpoint: GET /me/player/currently-playing (WORKING - not restricted)
    """
    try:
        sp = get_spotify_client(access_token)
        
        print(f"🎧 Fetching currently playing from Spotify...")
        playback = sp.current_playback()
        
        if not playback or not playback.get('is_playing'):
            return {
                'is_playing': False,
                'message': 'No track currently playing'
            }
        
        track = playback['item']
        device = playback['device']
        
        return {
            'is_playing': True,
            'track': {
                'id': track['id'],
                'name': track['name'],
                'artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name']
                    } for artist in track['artists']
                ],
                'album': {
                    'name': track['album']['name'],
                    'images': track['album']['images']
                },
                'duration_ms': track['duration_ms'],
                'popularity': track.get('popularity', 0),
                'external_url': track['external_urls']['spotify'],
                'uri': track['uri']
            },
            'device': {
                'name': device.get('name'),
                'type': device.get('type'),
                'volume_percent': device.get('volume_percent')
            },
            'progress_ms': playback.get('progress_ms', 0),
            'shuffle_state': playback.get('shuffle_state', False),
            'repeat_state': playback.get('repeat_state', 'off'),
            'timestamp': playback.get('timestamp')
        }
        
    except Exception as e:
        print(f"❌ Error getting currently playing: {e}")
        return None


async def get_user_top_tracks(
    access_token: str,
    time_range: str = 'medium_term',
    limit: int = 20
) -> List[Dict]:
    """
    Get user's top tracks
    
    Endpoint: GET /me/top/tracks (WORKING - not restricted)
    Requires scope: user-top-read
    """
    cache_key = f"spotify:top_tracks:{access_token[:10]}:{time_range}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        
        print(f"🎵 Fetching user's top tracks from Spotify...")
        results = sp.current_user_top_tracks(
            limit=limit,
            time_range=time_range
        )
        
        tracks = []
        for track in results['items']:
            tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name']
                    } for artist in track['artists']
                ],
                'album': {
                    'name': track['album']['name'],
                    'images': track['album']['images']
                },
                'popularity': track.get('popularity', 0),
                'external_url': track['external_urls']['spotify']
            })
        
        # Cache for 1 hour
        await cache_service.set_in_cache(cache_key, tracks, expiration=3600)
        
        print(f"✅ Retrieved {len(tracks)} top tracks")
        return tracks
        
    except Exception as e:
        print(f"❌ Error getting top tracks: {e}")
        return []


async def get_user_top_artists(
    access_token: str,
    time_range: str = 'medium_term',
    limit: int = 20
) -> List[Dict]:
    """
    Get user's top artists
    
    Endpoint: GET /me/top/artists (WORKING - not restricted)
    Requires scope: user-top-read
    """
    cache_key = f"spotify:top_artists:{access_token[:10]}:{time_range}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        
        print(f"🎸 Fetching user's top artists from Spotify...")
        results = sp.current_user_top_artists(
            limit=limit,
            time_range=time_range
        )
        
        artists = []
        for artist in results['items']:
            artists.append({
                'id': artist['id'],
                'name': artist['name'],
                'genres': artist.get('genres', []),
                'popularity': artist.get('popularity', 0),
                'followers': artist['followers'].get('total', 0),
                'images': artist.get('images', []),
                'external_url': artist['external_urls']['spotify']
            })
        
        # Cache for 1 hour
        await cache_service.set_in_cache(cache_key, artists, expiration=3600)
        
        print(f"✅ Retrieved {len(artists)} top artists")
        return artists
        
    except Exception as e:
        print(f"❌ Error getting top artists: {e}")
        return []


async def get_recently_played(
    access_token: str,
    limit: int = 50
) -> List[Dict]:
    """
    Get recently played tracks
    
    Endpoint: GET /me/player/recently-played (WORKING - not restricted)
    Requires scope: user-read-recently-played
    """
    cache_key = f"spotify:recently_played:{access_token[:10]}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        
        print(f"⏮️ Fetching recently played tracks...")
        results = sp.current_user_recently_played(limit=limit)
        
        tracks = []
        for item in results['items']:
            track = item['track']
            tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name']
                    } for artist in track['artists']
                ],
                'album': {
                    'name': track['album']['name'],
                    'images': track['album']['images']
                },
                'played_at': item['played_at'],
                'context': item.get('context'),
                'external_url': track['external_urls']['spotify']
            })
        
        # Cache for 2 minutes (recent data changes frequently)
        await cache_service.set_in_cache(cache_key, tracks, expiration=120)
        
        print(f"✅ Retrieved {len(tracks)} recently played tracks")
        return tracks
        
    except Exception as e:
        print(f"❌ Error getting recently played: {e}")
        return []


async def get_saved_tracks(
    access_token: str,
    limit: int = 50
) -> List[Dict]:
    """
    Get user's saved/liked tracks
    
    Endpoint: GET /me/tracks (WORKING - not restricted)
    Requires scope: user-library-read
    """
    try:
        sp = get_spotify_client(access_token)
        
        print(f"💚 Fetching saved tracks...")
        results = sp.current_user_saved_tracks(limit=limit)
        
        tracks = []
        for item in results['items']:
            track = item['track']
            tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name']
                    } for artist in track['artists']
                ],
                'album': {
                    'name': track['album']['name'],
                    'images': track['album']['images']
                },
                'added_at': item['added_at'],
                'popularity': track.get('popularity', 0),
                'external_url': track['external_urls']['spotify']
            })
        
        print(f"✅ Retrieved {len(tracks)} saved tracks")
        return tracks
        
    except Exception as e:
        print(f"❌ Error getting saved tracks: {e}")
        return []


async def search_tracks(
    query: str,
    access_token: Optional[str] = None,
    limit: int = 20
) -> List[Dict]:
    """
    Search for tracks on Spotify
    
    Endpoint: GET /search (WORKING - not restricted)
    """
    cache_key = f"spotify:search:{query}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        
        print(f"🔍 Searching Spotify for: {query}")
        results = sp.search(q=query, type='track', limit=limit)
        
        tracks = []
        for track in results['tracks']['items']:
            tracks.append({
                'id': track['id'],
                'name': track['name'],
                'artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name']
                    } for artist in track['artists']
                ],
                'album': {
                    'name': track['album']['name'],
                    'images': track['album']['images']
                },
                'popularity': track.get('popularity', 0),
                'duration_ms': track['duration_ms'],
                'preview_url': track.get('preview_url'),
                'external_url': track['external_urls']['spotify']
            })
        
        # Cache for 1 hour
        await cache_service.set_in_cache(cache_key, tracks, expiration=3600)
        
        print(f"✅ Found {len(tracks)} tracks on Spotify")
        return tracks
        
    except Exception as e:
        print(f"❌ Error searching tracks: {e}")
        return []


async def batch_get_tracks(
    track_ids: List[str],
    access_token: Optional[str] = None
) -> List[Dict]:
    """
    Get multiple tracks in batch
    
    Endpoint: GET /tracks (WORKING - not restricted)
    """
    if not track_ids:
        return []
    
    try:
        sp = get_spotify_client(access_token)
        
        print(f"📦 Batch fetching {len(track_ids)} tracks from Spotify...")
        
        # Spotify allows max 50 tracks per request
        all_tracks = []
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            results = sp.tracks(batch)
            
            for track in results['tracks']:
                if not track:
                    continue
                
                all_tracks.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [
                        {
                            'id': artist['id'],
                            'name': artist['name']
                        } for artist in track['artists']
                    ],
                    'album': {
                        'name': track['album']['name'],
                        'images': track['album']['images']
                    },
                    'duration_ms': track['duration_ms'],
                    'popularity': track.get('popularity', 0),
                    'external_url': track['external_urls']['spotify']
                })
        
        print(f"✅ Batch retrieved {len(all_tracks)} tracks")
        return all_tracks
        
    except Exception as e:
        print(f"❌ Error batch getting tracks: {e}")
        return []


# ============================================
# Helper Functions
# ============================================

def extract_track_artists_names(track: Dict) -> List[str]:
    """Extract artist names from track dict"""
    if 'artists' in track:
        return [artist['name'] for artist in track['artists']]
    return []


def get_primary_artist_name(track: Dict) -> str:
    """Get primary artist name"""
    if 'artists' in track and track['artists']:
        return track['artists'][0]['name']
    return "Unknown Artist"


async def test_spotify_service(access_token: str):
    """Test Spotify service endpoints"""
    print("\n" + "="*60)
    print("🧪 Testing Spotify Service (Hybrid Approach)")
    print("="*60)
    
    print("\n1. Testing user playlists...")
    playlists = await get_user_playlists(access_token, limit=5)
    print(f"   Found {len(playlists)} playlists")
    
    print("\n2. Testing currently playing...")
    current = await get_currently_playing(access_token)
    if current and current.get('is_playing'):
        print(f"   Now playing: {current['track']['name']}")
    
    print("\n3. Testing top tracks...")
    top_tracks = await get_user_top_tracks(access_token, limit=5)
    print(f"   Found {len(top_tracks)} top tracks")
    
    print("\n4. Testing search...")
    results = await search_tracks("Happy", limit=3)
    print(f"   Found {len(results)} results")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)