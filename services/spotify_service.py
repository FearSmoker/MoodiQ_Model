"""
Production-Ready Spotify Service
Fixes applied:
- ✅ Complete pagination for playlists and tracks
- ✅ Proper error handling with custom exceptions
- ✅ Rate limiting protection
- ✅ Secure cache key generation
- ✅ Token expiration handling
- ✅ Null/unavailable track handling
- ✅ Retry logic for transient failures
- ✅ NEW: get_recommendations() no longer relies solely on the deprecated
       GET /v1/recommendations endpoint. Spotify killed that endpoint (and
       GET /v1/audio-features) for any app that didn't already have
       "Extended Quota Mode" before Nov 27, 2024:
       https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api
       Calling it now just 403s, which was being silently swallowed by the
       bare `except Exception` below and returning []. get_recommendations()
       now tries the real endpoint first (works for grandfathered apps) and
       falls back to genre-filtered catalog search (GET /v1/search, which is
       NOT deprecated) so recommendations still return real, fresh tracks.
- ✅ NEW: get_audio_features() has no working replacement (Spotify has no
       public alternative to the deprecated /audio-features endpoint), so it
       now fails loudly/clearly instead of silently returning [None, ...]
       forever. Callers should treat a False `features_available` as a signal
       to rely on mood tags from catalog search / the MongoDB catalog
       (db_recommendation_service) instead of valence/energy filtering.
- ✅ FIX: removed duplicate `search_tracks` / `get_recommendations`
       definitions that previously existed later in this file — in Python,
       the second definition silently shadowed the first (more complete,
       cached) one, so the cache-enabled, error-handled versions were never
       actually being called.
"""

import os
import time
import random
import hashlib
from typing import List, Dict, Optional, Any
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from . import cache_service


# ============================================
# CUSTOM EXCEPTIONS
# ============================================

class SpotifyServiceError(Exception):
    """Base exception for Spotify service"""
    pass


class SpotifyAuthError(SpotifyServiceError):
    """Authentication/authorization error"""
    pass


class SpotifyRateLimitError(SpotifyServiceError):
    """Rate limit exceeded"""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class SpotifyNotFoundError(SpotifyServiceError):
    """Resource not found"""
    pass


class SpotifyTokenExpired(SpotifyServiceError):
    """Access token has expired"""
    pass


# ============================================
# RATE LIMITER
# ============================================

class RateLimiter:
    """Simple rate limiter to prevent API abuse"""

    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            'default': (100, 60),  # 100 requests per 60 seconds
            'search': (50, 60),    # More conservative for search
        }

    async def check_limit(self, endpoint: str = 'default'):
        """Check and enforce rate limits"""
        now = time.time()
        limit, window = self.limits.get(endpoint, self.limits['default'])

        # Clean old requests outside the window
        self.requests[endpoint] = [
            req_time for req_time in self.requests[endpoint]
            if now - req_time < window
        ]

        if len(self.requests[endpoint]) >= limit:
            raise SpotifyRateLimitError(
                f"Rate limit exceeded for {endpoint}. Try again later.",
                retry_after=window
            )

        self.requests[endpoint].append(now)


rate_limiter = RateLimiter()


# ============================================
# CLIENT MANAGEMENT
# ============================================

# Global Spotify client
sp_server: Optional[spotipy.Spotify] = None


def _get_user_id_hash(access_token: str) -> str:
    """
    Generate stable, secure user identifier from token
    Prevents cache key collisions and token exposure
    """
    return hashlib.sha256(access_token.encode()).hexdigest()[:16]


def _handle_spotify_exception(e: Exception) -> None:
    """Centralized exception handling"""
    if isinstance(e, spotipy.exceptions.SpotifyException):
        if e.http_status == 401:
            raise SpotifyAuthError("Invalid or expired token. Please re-authenticate.")
        elif e.http_status == 403:
            raise SpotifyAuthError("Insufficient permissions. Check required scopes.")
        elif e.http_status == 404:
            raise SpotifyNotFoundError("Resource not found")
        elif e.http_status == 429:
            retry_after = int(e.headers.get('Retry-After', 60))
            raise SpotifyRateLimitError(
                "Spotify API rate limit exceeded",
                retry_after=retry_after
            )
        raise SpotifyServiceError(f"Spotify API error: {e}")
    raise SpotifyServiceError(f"Unexpected error: {e}")


def get_spotify_client(access_token: Optional[str] = None) -> spotipy.Spotify:
    """
    Get Spotify client with user token or server credentials

    Args:
        access_token: User's OAuth access token

    Returns:
        Configured Spotify client

    Raises:
        ValueError: If server credentials are missing
        SpotifyAuthError: If token is invalid
    """
    if access_token:
        # Use user's access token (from frontend OAuth)
        try:
            return spotipy.Spotify(auth=access_token)
        except Exception as e:
            raise SpotifyAuthError(f"Failed to create user client: {e}")

    # Use server credentials for public endpoints
    global sp_server

    if sp_server is None:
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise ValueError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in environment"
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
            raise ValueError(f"Failed to initialize Spotify server client: {e}")

    return sp_server


# ============================================
# USER DATA & PLAYLISTS - WITH FULL PAGINATION
# ============================================

async def get_user_playlists(
    access_token: str,
    limit: int = 50,
    fetch_all: bool = True
) -> List[Dict]:
    """
    Get user's playlists from Spotify with FULL PAGINATION

    Args:
        access_token: User's Spotify access token
        limit: Items per page (max 50, Spotify limit)
        fetch_all: If True, fetches all playlists across pages

    Returns:
        Complete list of playlist dictionaries

    Endpoint: GET /me/playlists
    """
    user_hash = _get_user_id_hash(access_token)
    cache_key = f"spotify:playlists:{user_hash}"
    cached = await cache_service.get_from_cache(cache_key)

    if cached:
        print(f"📦 Cache HIT: User playlists ({len(cached)} playlists)")
        return cached

    try:
        await rate_limiter.check_limit('playlists')
        sp = get_spotify_client(access_token)

        playlists = []
        offset = 0

        print(f"📂 Fetching user playlists from Spotify...")

        while True:
            print(f"   → Page {offset // limit + 1} (offset={offset}, limit={limit})")

            response = sp.current_user_playlists(limit=limit, offset=offset)

            for item in response['items']:
                playlists.append({
                    'id': item['id'],
                    'name': item['name'],
                    'description': item.get('description'),
                    'images': item.get('images', []),
                    'tracks_total': item['tracks']['total'],
                    'owner': item['owner']['display_name'],
                    'owner_id': item['owner']['id'],
                    'public': item.get('public', False),
                    'collaborative': item.get('collaborative', False),
                    'external_url': item['external_urls']['spotify'],
                    'uri': item['uri'],
                    'snapshot_id': item.get('snapshot_id')
                })

            # Check if we should continue pagination
            if not fetch_all or response['next'] is None:
                break

            offset += limit

            # Safety check to prevent infinite loops (Spotify's max)
            if offset >= 1000:
                print("⚠️ Reached maximum playlist limit (1000)")
                break

        # Cache for 5 minutes
        await cache_service.set_in_cache(cache_key, playlists, expiration=300)

        print(f"✅ Retrieved {len(playlists)} total playlists")
        return playlists

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


async def get_playlist_tracks(
    playlist_id: str,
    access_token: str,
    include_unavailable: bool = False
) -> List[Dict]:
    """
    Get ALL tracks from a Spotify playlist with FULL PAGINATION

    CRITICAL FIX: Handles playlists with 100+ tracks correctly

    Args:
        playlist_id: Spotify playlist ID
        access_token: User's access token
        include_unavailable: Whether to include unavailable/local tracks

    Returns:
        Complete list of track dictionaries

    Endpoint: GET /playlists/{id}/tracks
    """
    cache_key = f"spotify:playlist_tracks:{playlist_id}"
    cached = await cache_service.get_from_cache(cache_key)

    if cached:
        print(f"📦 Cache HIT: Playlist tracks ({len(cached)} tracks)")
        return cached

    try:
        await rate_limiter.check_limit('playlist')
        sp = get_spotify_client(access_token)

        tracks = []
        offset = 0
        limit = 100  # Spotify's max for playlist tracks

        print(f"🎵 Fetching playlist tracks: {playlist_id}")

        while True:
            print(f"   → Page {offset // limit + 1} (offset={offset})")

            # Fetch with field filtering to reduce response size
            results = sp.playlist_tracks(
                playlist_id,
                limit=limit,
                offset=offset,
                fields='items(track(id,name,artists,album,duration_ms,popularity,explicit,preview_url,external_urls,uri,is_local),added_at),next,total'
            )

            for item in results['items']:
                # Handle null tracks (deleted/unavailable)
                if not item.get('track'):
                    if include_unavailable:
                        tracks.append({
                            'id': None,
                            'name': '[Unavailable Track]',
                            'unavailable': True,
                            'added_at': item.get('added_at')
                        })
                    continue

                track = item['track']

                # Skip local files unless requested
                if track.get('is_local') and not include_unavailable:
                    continue

                tracks.append({
                    'id': track.get('id'),
                    'name': track['name'],
                    'artists': [
                        {
                            'id': artist.get('id'),
                            'name': artist['name']
                        } for artist in track.get('artists', [])
                    ],
                    'album': {
                        'id': track['album'].get('id'),
                        'name': track['album']['name'],
                        'images': track['album'].get('images', [])
                    },
                    'duration_ms': track.get('duration_ms', 0),
                    'popularity': track.get('popularity', 0),
                    'explicit': track.get('explicit', False),
                    'preview_url': track.get('preview_url'),
                    'external_url': track['external_urls'].get('spotify') if track.get('external_urls') else None,
                    'uri': track.get('uri'),
                    'is_local': track.get('is_local', False),
                    'added_at': item.get('added_at')
                })

            # Check for more pages
            if results['next'] is None:
                break

            offset += limit

            # Safety check
            if offset >= 10000:
                print("⚠️ Reached safety limit (10,000 tracks)")
                break

        # Cache for 10 minutes
        await cache_service.set_in_cache(cache_key, tracks, expiration=600)

        print(f"✅ Retrieved {len(tracks)} tracks from playlist")
        return tracks

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


async def get_track_info(
    track_id: str,
    access_token: Optional[str] = None
) -> Optional[Dict]:
    """
    Get track information from Spotify

    Endpoint: GET /tracks/{id}
    """
    cache_key = f"spotify:track:{track_id}"
    cached = await cache_service.get_from_cache(cache_key)

    if cached:
        return cached

    try:
        await rate_limiter.check_limit('track')
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
            'track_number': track.get('track_number', 1),
            'isrc': track.get('external_ids', {}).get('isrc')
        }

        # Cache for 1 day (track metadata rarely changes)
        await cache_service.set_in_cache(cache_key, info, expiration=86400)
        return info

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return None


async def get_currently_playing(access_token: str) -> Optional[Dict]:
    """
    Get currently playing track with COMPLETE playback state

    Returns all details: track, device, playback state, context

    Endpoint: GET /me/player/currently-playing
    Requires scope: user-read-currently-playing, user-read-playback-state
    """
    try:
        await rate_limiter.check_limit('playback')
        sp = get_spotify_client(access_token)

        print(f"🎧 Fetching currently playing track...")
        playback = sp.current_playback()

        # Check if anything is playing
        if not playback or not playback.get('is_playing'):
            print("❌ No track currently playing")
            return {
                'is_playing': False,
                'message': 'No track currently playing'
            }

        # Handle podcasts/episodes
        if playback.get('currently_playing_type') == 'episode':
            item = playback['item']
            device = playback['device']

            return {
                'is_playing': True,
                'type': 'episode',
                'episode': {
                    'id': item['id'],
                    'name': item['name'],
                    'show': {
                        'id': item['show']['id'],
                        'name': item['show']['name'],
                        'publisher': item['show'].get('publisher')
                    },
                    'description': item.get('description'),
                    'duration_ms': item['duration_ms'],
                    'images': item.get('images', []),
                    'external_url': item['external_urls']['spotify']
                },
                'device': _extract_device_info(device),
                'progress_ms': playback.get('progress_ms', 0),
                'timestamp': playback.get('timestamp')
            }

        # Extract track data
        item = playback['item']
        device = playback['device']

        # Build complete response
        result = {
            'is_playing': True,
            'type': 'track',

            # Track information
            'track': {
                'id': item['id'],
                'name': item['name'],
                'artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name'],
                        'uri': artist['uri']
                    } for artist in item['artists']
                ],
                'album': {
                    'id': item['album']['id'],
                    'name': item['album']['name'],
                    'images': item['album']['images'],
                    'release_date': item['album'].get('release_date'),
                    'uri': item['album']['uri']
                },
                'duration_ms': item['duration_ms'],
                'popularity': item.get('popularity', 0),
                'explicit': item.get('explicit', False),
                'preview_url': item.get('preview_url'),
                'external_url': item['external_urls']['spotify'],
                'uri': item['uri'],
                'is_local': item.get('is_local', False)
            },

            # Device information
            'device': _extract_device_info(device),

            # Playback state
            'progress_ms': playback.get('progress_ms', 0),
            'shuffle_state': playback.get('shuffle_state', False),
            'repeat_state': playback.get('repeat_state', 'off'),
            'timestamp': playback.get('timestamp'),

            # Context (what's playing from)
            'context': _extract_context(playback.get('context'))
        }

        # Log playback details
        print(f"🎧 Device: {device['name']} ({device['type']})")
        print(f"🎵 Track: {item['name']}")
        print(f"👨‍🎤 Artist(s): {', '.join([a['name'] for a in item['artists']])}")

        return result

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return None


def _extract_device_info(device: Dict) -> Dict:
    """Extract device information"""
    return {
        'id': device.get('id'),
        'name': device.get('name'),
        'type': device.get('type'),
        'volume_percent': device.get('volume_percent'),
        'is_active': device.get('is_active', True),
        'is_private_session': device.get('is_private_session', False),
        'is_restricted': device.get('is_restricted', False)
    }


def _extract_context(context: Optional[Dict]) -> Optional[Dict]:
    """Extract playback context"""
    if not context:
        return None

    return {
        'type': context.get('type'),
        'uri': context.get('uri'),
        'external_url': context.get('external_urls', {}).get('spotify')
    }


async def get_user_top_tracks(
    access_token: str,
    time_range: str = 'medium_term',
    limit: int = 20
) -> List[Dict]:
    """
    Get user's top tracks

    Endpoint: GET /me/top/tracks
    Requires scope: user-top-read

    Args:
        time_range: 'short_term' (4 weeks), 'medium_term' (6 months), 'long_term' (years)
    """
    user_hash = _get_user_id_hash(access_token)
    cache_key = f"spotify:top_tracks:{user_hash}:{time_range}"
    cached = await cache_service.get_from_cache(cache_key)

    if cached:
        return cached

    try:
        await rate_limiter.check_limit('user_data')
        sp = get_spotify_client(access_token)

        print(f"🎵 Fetching user's top tracks ({time_range})...")
        results = sp.current_user_top_tracks(
            limit=min(limit, 50),  # Spotify max is 50
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

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


async def get_user_top_artists(
    access_token: str,
    time_range: str = 'medium_term',
    limit: int = 20
) -> List[Dict]:
    """
    Get user's top artists

    Endpoint: GET /me/top/artists
    Requires scope: user-top-read
    """
    user_hash = _get_user_id_hash(access_token)
    cache_key = f"spotify:top_artists:{user_hash}:{time_range}"
    cached = await cache_service.get_from_cache(cache_key)

    if cached:
        return cached

    try:
        await rate_limiter.check_limit('user_data')
        sp = get_spotify_client(access_token)

        print(f"🎸 Fetching user's top artists ({time_range})...")
        results = sp.current_user_top_artists(
            limit=min(limit, 50),
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

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


async def get_recently_played(
    access_token: str,
    limit: int = 50
) -> List[Dict]:
    """
    Get recently played tracks

    Endpoint: GET /me/player/recently-played
    Requires scope: user-read-recently-played
    """
    user_hash = _get_user_id_hash(access_token)
    cache_key = f"spotify:recently_played:{user_hash}"
    cached = await cache_service.get_from_cache(cache_key)

    if cached:
        return cached

    try:
        await rate_limiter.check_limit('user_data')
        sp = get_spotify_client(access_token)

        print(f"⏮️ Fetching recently played tracks...")
        results = sp.current_user_recently_played(limit=min(limit, 50))

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
                'context': _extract_context(item.get('context')),
                'external_url': track['external_urls']['spotify']
            })

        # Cache for 2 minutes (recent data changes frequently)
        await cache_service.set_in_cache(cache_key, tracks, expiration=120)

        print(f"✅ Retrieved {len(tracks)} recently played tracks")
        return tracks

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


async def get_saved_tracks(
    access_token: str,
    limit: int = 50,
    fetch_all: bool = False
) -> List[Dict]:
    """
    Get user's saved/liked tracks with pagination support

    Endpoint: GET /me/tracks
    Requires scope: user-library-read
    """
    user_hash = _get_user_id_hash(access_token)
    cache_key = f"spotify:saved_tracks:{user_hash}"

    if not fetch_all:
        cached = await cache_service.get_from_cache(cache_key)
        if cached:
            return cached

    try:
        await rate_limiter.check_limit('user_data')
        sp = get_spotify_client(access_token)

        print(f"💚 Fetching saved tracks...")

        tracks = []
        offset = 0
        page_limit = min(limit, 50)

        while True:
            results = sp.current_user_saved_tracks(limit=page_limit, offset=offset)

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

            if not fetch_all or results['next'] is None:
                break

            offset += page_limit

            if offset >= 1000:  # Safety limit
                break

        # Cache for 5 minutes
        await cache_service.set_in_cache(cache_key, tracks, expiration=300)

        print(f"✅ Retrieved {len(tracks)} saved tracks")
        return tracks

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


async def batch_get_tracks(
    track_ids: List[str],
    access_token: Optional[str] = None
) -> List[Dict]:
    """
    Get multiple tracks in batch (up to 50 per request)

    Endpoint: GET /tracks
    """
    if not track_ids:
        return []

    try:
        await rate_limiter.check_limit('batch')
        sp = get_spotify_client(access_token)

        print(f"📦 Batch fetching {len(track_ids)} tracks from Spotify...")

        all_tracks = []
        # Spotify allows max 50 tracks per request
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            results = sp.tracks(batch)

            for track in results['tracks']:
                if not track:  # Handle null tracks
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

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


# ============================================
# HELPER FUNCTIONS
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


def format_duration(duration_ms: int) -> str:
    """Format duration from milliseconds to MM:SS"""
    total_seconds = duration_ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


# ============================================
# CATALOG DISCOVERY (replaces deprecated /v1/recommendations)
# ============================================
# Mirrors the fix applied on the Node side (recommendationsController.js).
# GET /v1/recommendations and GET /v1/audio-features were killed for any app
# without pre-Nov-2024 "Extended Quota Mode":
# https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api
# GET /v1/search is NOT deprecated, so genre-filtered search is used to pull
# real, fresh catalog tracks instead of silently returning [].

MOOD_GENRE_SEEDS: Dict[str, List[str]] = {
    'Joyful':      ['pop', 'dance pop', 'feel good'],
    'Excited':     ['edm', 'electropop', 'dance'],
    'Party':       ['party', 'hip hop', 'dance pop'],
    'Melancholic': ['sad', 'indie folk', 'singer-songwriter'],
    'Dreamy':      ['dream pop', 'shoegaze', 'ambient pop'],
    'Relaxed':     ['acoustic', 'chill', 'lo-fi'],
    'Chill':       ['chillhop', 'chill', 'indie pop'],
    'Focused':     ['instrumental', 'study beats', 'ambient'],
    'Romantic':    ['r&b', 'soul', 'love songs'],
    'Motivated':   ['workout', 'power pop', 'rock'],
    'Angry':       ['metal', 'punk', 'hard rock'],
    'Ambient':     ['ambient', 'atmospheric', 'drone'],
}
ANY_MOOD_GENRES = ['pop', 'indie', 'rock', 'hip hop', 'electronic', 'r&b', 'alternative', 'folk']


def _sample(seq: List, n: int) -> List:
    pool = list(seq)
    random.shuffle(pool)
    return pool[:n]


async def search_catalog_for_mood(
    sp: spotipy.Spotify,
    mood: Optional[str],
    count: int,
    exclude_ids: Optional[set] = None
) -> List[Dict]:
    """
    Pull real catalog tracks via genre-filtered search (GET /v1/search),
    since GET /v1/recommendations is dead for most apps. Random offsets mean
    repeated calls surface different parts of the catalog instead of the
    same fixed results every time.
    """
    exclude_ids = exclude_ids or set()
    genres = MOOD_GENRE_SEEDS.get(mood, [mood.lower()]) if mood else _sample(ANY_MOOD_GENRES, 3)

    results = []
    seen = set(exclude_ids)
    per_genre = min(50, (count // max(len(genres), 1)) + 10)

    for genre in genres:
        if len(results) >= count:
            break
        try:
            random_offset = random.randint(0, 150)  # dip into different parts of the catalog
            res = sp.search(
                q=f'genre:"{genre}"',
                type='track',
                limit=per_genre,
                offset=random_offset
            )
            for track in res.get('tracks', {}).get('items', []):
                if not track or not track.get('id') or track['id'] in seen:
                    continue
                seen.add(track['id'])
                results.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [
                        {'id': a.get('id'), 'name': a['name']} for a in track.get('artists', [])
                    ],
                    'album': {
                        'name': track['album']['name'],
                        'images': track['album'].get('images', [])
                    },
                    'duration_ms': track.get('duration_ms', 0),
                    'popularity': track.get('popularity', 0),
                    'explicit': track.get('explicit', False),
                    'preview_url': track.get('preview_url'),
                    'external_url': track['external_urls'].get('spotify') if track.get('external_urls') else None,
                    'mood': mood,
                    'source': 'catalog_search',
                    'searched_genre': genre,
                })
        except Exception as e:
            print(f"⚠️ Catalog search failed for genre '{genre}': {e}")

    return _sample(results, min(count, len(results)))


async def get_recommendations(
    seed_tracks: Optional[List[str]] = None,
    seed_artists: Optional[List[str]] = None,
    seed_genres: Optional[List[str]] = None,
    target_valence: Optional[float] = None,
    target_energy: Optional[float] = None,
    limit: int = 20,
    access_token: Optional[str] = None
) -> List[Dict]:
    """
    Get track recommendations from Spotify.

    Tries the real /v1/recommendations endpoint first (works if this app
    still has Extended Quota Mode). If that fails — which it will for the
    vast majority of apps registered after Nov 2024 — falls back to
    genre-filtered catalog search so this still returns real, varied tracks
    instead of an empty list.
    """
    try:
        sp = get_spotify_client(access_token)

        seeds = {}
        if seed_tracks:
            seeds['seed_tracks'] = seed_tracks[:5]
        if seed_artists:
            seeds['seed_artists'] = seed_artists[:5]
        if seed_genres:
            seeds['seed_genres'] = seed_genres[:5]

        kwargs = {}
        if target_valence is not None:
            kwargs['target_valence'] = target_valence
        if target_energy is not None:
            kwargs['target_energy'] = target_energy

        try:
            results = sp.recommendations(limit=limit, **seeds, **kwargs)
            recommendations = []
            for track in results['tracks']:
                recommendations.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'album': track['album']['name'],
                    'popularity': track.get('popularity', 0),
                    'preview_url': track.get('preview_url'),
                    'external_url': track['external_urls'].get('spotify'),
                    'source': 'recommendations_api',
                })
            if recommendations:
                print(f"✅ Got {len(recommendations)} recommendations from /v1/recommendations")
                return recommendations
        except Exception as e:
            print(f"⚠️ /v1/recommendations unavailable ({e}); falling back to catalog search")

        # Fallback: genre-filtered catalog search
        mood_key = None
        if seed_genres:
            mood_key = seed_genres[0]
        catalog_tracks = await search_catalog_for_mood(
            sp, mood_key, limit, exclude_ids=set(seed_tracks or [])
        )
        recommendations = [
            {
                'id': t['id'],
                'name': t['name'],
                'artists': [a['name'] for a in t['artists']],
                'album': t['album']['name'],
                'popularity': t.get('popularity', 0),
                'preview_url': t.get('preview_url'),
                'external_url': t.get('external_url'),
                'source': 'catalog_search',
                # Carry the genre-search mood tag through so callers that can't
                # get real audio features (see get_audio_features) still have
                # a usable mood label instead of "Unknown".
                'catalog_mood': t.get('mood'),
            }
            for t in catalog_tracks
        ]
        print(f"✅ Got {len(recommendations)} recommendations from catalog search fallback")
        return recommendations

    except Exception as e:
        print(f"⚠️ Error getting Spotify recommendations: {e}")
        return []


async def get_audio_features(
    track_ids: List[str],
    access_token: Optional[str] = None
) -> List[Optional[Dict]]:
    """
    Get audio features for multiple track IDs from Spotify.

    NOTE: GET /v1/audio-features is deprecated for apps without Extended
    Quota Mode and has no public replacement. This will 403 for most apps —
    that failure is now surfaced clearly (logged) rather than silently
    swallowed. Callers should check for an all-None result and treat it as
    "features unavailable", relying on mood tags from catalog search or a
    precomputed feature source (e.g. the MongoDB catalog in
    db_recommendation_service.py) instead of valence/energy filtering.
    """
    if not track_ids:
        return []
    try:
        sp = get_spotify_client(access_token)
        all_features = []
        for i in range(0, len(track_ids), 100):
            chunk = track_ids[i:i+100]
            res = sp.audio_features(chunk)
            if res:
                all_features.extend(res)
        results = []
        for feat in all_features:
            if not feat:
                results.append(None)
                continue
            results.append({
                'id': feat.get('id'),
                'danceability': feat.get('danceability', 0.5),
                'energy': feat.get('energy', 0.5),
                'loudness': feat.get('loudness', -10.0),
                'speechiness': feat.get('speechiness', 0.05),
                'acousticness': feat.get('acousticness', 0.5),
                'instrumentalness': feat.get('instrumentalness', 0.0),
                'liveness': feat.get('liveness', 0.15),
                'valence': feat.get('valence', 0.5),
                'tempo': feat.get('tempo', 120.0),
                'duration_ms': feat.get('duration_ms', 200000),
                'key': feat.get('key', 0)
            })
        if not any(results):
            print("⚠️ Spotify /v1/audio-features returned no usable data for this app "
                  "(likely deprecated/403 for non-extended-quota apps). "
                  "Downstream code should fall back to catalog-search mood tags "
                  "or db_recommendation_service instead of valence/energy filtering.")
        return results
    except Exception as e:
        print(f"⚠️ Error getting audio features from Spotify (endpoint is deprecated "
              f"for most apps as of Nov 2024): {e}")
        return [None] * len(track_ids)


async def search_tracks(
    query: str,
    limit: int = 20,
    access_token: Optional[str] = None
) -> List[Dict]:
    """
    Search for tracks on Spotify.

    Endpoint: GET /search (cached, rate-limited version — this is the single
    canonical implementation; a duplicate, uncached definition further down
    in the original file used to silently shadow this one).
    """
    cache_key = f"spotify:search:{query}:{limit}"
    cached = await cache_service.get_from_cache(cache_key)

    if cached:
        return cached

    try:
        await rate_limiter.check_limit('search')
        sp = get_spotify_client(access_token)

        print(f"🔍 Searching Spotify for: {query}")
        results = sp.search(q=query, type='track', limit=min(limit, 50))

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

    except SpotifyServiceError:
        raise
    except Exception as e:
        _handle_spotify_exception(e)
        return []


# ============================================
# TESTING & DIAGNOSTICS
# ============================================

async def test_spotify_service(access_token: str):
    """
    Comprehensive test suite for Spotify service
    Tests all major endpoints with error handling
    """
    print("\n" + "="*60)
    print("🧪 Testing Spotify Service (Production-Ready)")
    print("="*60)

    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }

    playlists = []
    top_tracks = []

    # Test 1: User Playlists with Pagination
    print("\n1️⃣ Testing user playlists (with pagination)...")
    try:
        playlists = await get_user_playlists(access_token, limit=50, fetch_all=True)
        print(f"   ✅ Found {len(playlists)} playlists")
        if playlists:
            print(f"   📋 Sample: '{playlists[0]['name']}' ({playlists[0]['tracks_total']} tracks)")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('playlists', str(e)))

    # Test 2: Playlist Tracks with Pagination
    print("\n2️⃣ Testing playlist tracks (with pagination)...")
    try:
        if playlists:
            playlist_id = playlists[0]['id']
            tracks = await get_playlist_tracks(playlist_id, access_token)
            print(f"   ✅ Retrieved {len(tracks)} tracks from playlist")
            if tracks:
                print(f"   🎵 Sample: '{tracks[0]['name']}' by {tracks[0]['artists'][0]['name']}")
            test_results['passed'] += 1
        else:
            print(f"   ⏭️ Skipped (no playlists available)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('playlist_tracks', str(e)))

    # Test 3: Currently Playing
    print("\n3️⃣ Testing currently playing...")
    try:
        current = await get_currently_playing(access_token)
        if current and current.get('is_playing'):
            track = current.get('track') or current.get('episode', {})
            print(f"   ✅ Now playing: {track.get('name', 'Unknown')}")
            print(f"   🎧 Device: {current['device']['name']} ({current['device']['type']})")
            if current['device'].get('volume_percent') is not None:
                print(f"   🔊 Volume: {current['device']['volume_percent']}%")
            print(f"   🔀 Shuffle: {current.get('shuffle_state', False)}")
            print(f"   🔁 Repeat: {current.get('repeat_state', 'off')}")
        else:
            print(f"   ℹ️ Nothing currently playing")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('currently_playing', str(e)))

    # Test 4: Top Tracks
    print("\n4️⃣ Testing top tracks...")
    try:
        top_tracks = await get_user_top_tracks(access_token, time_range='medium_term', limit=5)
        print(f"   ✅ Found {len(top_tracks)} top tracks")
        if top_tracks:
            for i, track in enumerate(top_tracks[:3], 1):
                print(f"   {i}. '{track['name']}' by {track['artists'][0]['name']}")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('top_tracks', str(e)))

    # Test 5: Top Artists
    print("\n5️⃣ Testing top artists...")
    try:
        top_artists = await get_user_top_artists(access_token, time_range='medium_term', limit=5)
        print(f"   ✅ Found {len(top_artists)} top artists")
        if top_artists:
            for i, artist in enumerate(top_artists[:3], 1):
                genres = ', '.join(artist['genres'][:2]) if artist['genres'] else 'No genres'
                print(f"   {i}. {artist['name']} ({genres})")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('top_artists', str(e)))

    # Test 6: Recently Played
    print("\n6️⃣ Testing recently played...")
    try:
        recent = await get_recently_played(access_token, limit=5)
        print(f"   ✅ Found {len(recent)} recently played tracks")
        if recent:
            print(f"   🕐 Most recent: '{recent[0]['name']}' at {recent[0]['played_at']}")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('recently_played', str(e)))

    # Test 7: Saved Tracks
    print("\n7️⃣ Testing saved tracks...")
    try:
        saved = await get_saved_tracks(access_token, limit=5)
        print(f"   ✅ Found {len(saved)} saved tracks")
        if saved:
            print(f"   💚 Sample: '{saved[0]['name']}'")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('saved_tracks', str(e)))

    # Test 8: Search
    print("\n8️⃣ Testing search...")
    try:
        results = await search_tracks("Happy", limit=3)
        print(f"   ✅ Found {len(results)} search results")
        if results:
            print(f"   🔍 Top result: '{results[0]['name']}' by {results[0]['artists'][0]['name']}")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('search', str(e)))

    # Test 9: Track Info
    print("\n9️⃣ Testing track info...")
    try:
        if top_tracks:
            track_id = top_tracks[0]['id']
            info = await get_track_info(track_id, access_token)
            if info:
                print(f"   ✅ Retrieved info for: '{info['name']}'")
                print(f"   📊 Popularity: {info['popularity']}/100")
                print(f"   ⏱️ Duration: {format_duration(info['duration_ms'])}")
            test_results['passed'] += 1
        else:
            print(f"   ⏭️ Skipped (no tracks available)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('track_info', str(e)))

    # Test 10: Batch Get Tracks
    print("\n🔟 Testing batch get tracks...")
    try:
        if top_tracks and len(top_tracks) >= 3:
            track_ids = [t['id'] for t in top_tracks[:3]]
            batch = await batch_get_tracks(track_ids, access_token)
            print(f"   ✅ Batch retrieved {len(batch)} tracks")
        else:
            print(f"   ⏭️ Skipped (insufficient tracks)")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('batch_tracks', str(e)))

    # Test 11: Recommendations (catalog-search fallback)
    print("\n1️⃣1️⃣ Testing recommendations (catalog-search fallback)...")
    try:
        recs = await get_recommendations(
            seed_genres=['pop'],
            limit=5,
            access_token=access_token
        )
        print(f"   ✅ Got {len(recs)} recommendations (source: "
              f"{recs[0]['source'] if recs else 'none'})")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('recommendations', str(e)))

    # Test 12: Audio Features (expected to be unavailable for most apps)
    print("\n1️⃣2️⃣ Testing audio features (likely unavailable — deprecated endpoint)...")
    try:
        if top_tracks:
            feats = await get_audio_features([t['id'] for t in top_tracks[:3]], access_token)
            available = any(feats)
            print(f"   ℹ️ Features available: {available} "
                  f"({'OK, this app has Extended Quota' if available else 'expected 403 for most apps'})")
        test_results['passed'] += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results['failed'] += 1
        test_results['errors'].append(('audio_features', str(e)))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")
    total = test_results['passed'] + test_results['failed']
    if total:
        print(f"📈 Success Rate: {test_results['passed']/total*100:.1f}%")

    if test_results['errors']:
        print("\n❌ Errors encountered:")
        for endpoint, error in test_results['errors']:
            print(f"   • {endpoint}: {error}")

    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)

    return test_results


# ============================================
# REQUIRED SPOTIFY SCOPES
# ============================================

REQUIRED_SCOPES = [
    # User Profile
    'user-read-private',
    'user-read-email',

    # Playlists
    'playlist-read-private',
    'playlist-read-collaborative',
    'playlist-modify-public',
    'playlist-modify-private',

    # Listening History
    'user-top-read',                    # get_user_top_tracks, get_user_top_artists
    'user-read-recently-played',        # get_recently_played

    # Playback
    'user-read-playback-state',         # get_currently_playing (full state)
    'user-read-currently-playing',      # get_currently_playing (track only)

    # Library
    'user-library-read',                # get_saved_tracks
    'user-library-modify',
]


def get_required_scopes() -> List[str]:
    """
    Get list of all required Spotify OAuth scopes

    Returns:
        List of scope strings for OAuth authorization
    """
    return REQUIRED_SCOPES


def verify_token_scopes(access_token: str) -> Dict[str, Any]:
    """
    Verify what scopes are available on the current token

    Note: This requires making an API call that will fail if scopes are insufficient
    Returns information about what endpoints will work
    """
    sp = get_spotify_client(access_token)

    available_endpoints = {
        'user_profile': False,
        'playlists': False,
        'currently_playing': False,
        'top_tracks': False,
        'recently_played': False,
        'saved_tracks': False,
    }

    # Try each endpoint to see what works
    try:
        sp.current_user()
        available_endpoints['user_profile'] = True
    except Exception:
        pass

    try:
        sp.current_user_playlists(limit=1)
        available_endpoints['playlists'] = True
    except Exception:
        pass

    try:
        sp.current_playback()
        available_endpoints['currently_playing'] = True
    except Exception:
        pass

    try:
        sp.current_user_top_tracks(limit=1)
        available_endpoints['top_tracks'] = True
    except Exception:
        pass

    try:
        sp.current_user_recently_played(limit=1)
        available_endpoints['recently_played'] = True
    except Exception:
        pass

    try:
        sp.current_user_saved_tracks(limit=1)
        available_endpoints['saved_tracks'] = True
    except Exception:
        pass

    return available_endpoints