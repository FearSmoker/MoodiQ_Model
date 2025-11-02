"""
Spotify API service for Moodify-AI.

Handles:
- Audio features fetching
- Track information retrieval
- Artist information
- Playlist operations
- Smart caching to reduce API calls
"""

import os
from typing import List, Dict, Optional, Any
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from . import cache_service


# Global Spotify client for server-to-server calls
sp_server: Optional[spotipy.Spotify] = None


def get_spotify_client(access_token: Optional[str] = None) -> spotipy.Spotify:
    """
    Get Spotify client. Uses user token if provided, otherwise uses server credentials.
    
    Args:
        access_token: User's Spotify access token (optional)
        
    Returns:
        Configured Spotify client
    """
    if access_token:
        # Create client with user's access token
        return spotipy.Spotify(auth=access_token)
    
    # Use server-to-server authentication
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
            raise
    
    return sp_server


async def get_audio_features(
    track_ids: List[str], 
    access_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get audio features for multiple tracks with caching.
    
    Args:
        track_ids: List of Spotify track IDs
        access_token: User's access token (optional)
        
    Returns:
        List of audio feature dictionaries
    """
    if not track_ids:
        return []
    
    print(f"🎹 Fetching audio features for {len(track_ids)} tracks...")
    
    sp = get_spotify_client(access_token)
    
    features_list = []
    cached_features = {}
    tracks_to_fetch = []
    
    # Check cache for each track
    for track_id in track_ids:
        cached = await cache_service.get_cached_audio_features(track_id)
        
        if cached:
            cached_features[track_id] = cached
        else:
            tracks_to_fetch.append(track_id)
    
    print(f"📦 Cache hits: {len(cached_features)}, Cache misses: {len(tracks_to_fetch)}")
    
    # Fetch uncached tracks from Spotify
    if tracks_to_fetch:
        # Spotify allows max 100 tracks per request
        for i in range(0, len(tracks_to_fetch), 100):
            batch = tracks_to_fetch[i:i+100]
            
            try:
                batch_features = sp.audio_features(tracks=batch)
                
                # Process and cache results
                for features in batch_features:
                    if features and features.get('id'):
                        track_id = features['id']
                        
                        # Normalize features for consistency
                        normalized_features = normalize_audio_features(features)
                        
                        # Cache for 1 day (features don't change)
                        await cache_service.cache_audio_features(
                            track_id,
                            normalized_features
                        )
                        
                        cached_features[track_id] = normalized_features
                
            except spotipy.SpotifyException as e:
                print(f"⚠️ Spotify API error fetching features: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Error fetching audio features batch: {e}")
                continue
    
    # Assemble final list in original order
    for track_id in track_ids:
        if track_id in cached_features:
            features_list.append(cached_features[track_id])
        else:
            # Add placeholder for missing features
            print(f"⚠️ No features available for track: {track_id}")
            features_list.append(None)
    
    print(f"✅ Retrieved features for {len([f for f in features_list if f])} tracks")
    
    return features_list


def normalize_audio_features(features: Dict) -> Dict:
    """
    Normalize and validate audio features.
    
    Args:
        features: Raw features from Spotify API
        
    Returns:
        Normalized features dictionary
    """
    # Ensure all expected fields exist with defaults
    normalized = {
        'id': features.get('id'),
        'valence': float(features.get('valence', 0.5)),
        'energy': float(features.get('energy', 0.5)),
        'danceability': float(features.get('danceability', 0.5)),
        'acousticness': float(features.get('acousticness', 0.5)),
        'instrumentalness': float(features.get('instrumentalness', 0.0)),
        'speechiness': float(features.get('speechiness', 0.05)),
        'tempo': float(features.get('tempo', 120.0)),
        'loudness': float(features.get('loudness', -10.0)),
        'liveness': float(features.get('liveness', 0.1)),
        'key': int(features.get('key', 0)),
        'mode': int(features.get('mode', 1)),
        'time_signature': int(features.get('time_signature', 4)),
        'duration_ms': int(features.get('duration_ms', 0)),
    }
    
    # Validate ranges
    for key in ['valence', 'energy', 'danceability', 'acousticness', 
                'instrumentalness', 'speechiness', 'liveness']:
        normalized[key] = max(0.0, min(1.0, normalized[key]))
    
    normalized['tempo'] = max(0.0, min(300.0, normalized['tempo']))
    normalized['loudness'] = max(-60.0, min(0.0, normalized['loudness']))
    normalized['key'] = max(0, min(11, normalized['key']))
    normalized['mode'] = max(0, min(1, normalized['mode']))
    normalized['time_signature'] = max(3, min(7, normalized['time_signature']))
    
    return normalized


async def get_track_info(
    track_id: str,
    access_token: Optional[str] = None
) -> Optional[Dict]:
    """
    Get detailed track information.
    
    Args:
        track_id: Spotify track ID
        access_token: User's access token (optional)
        
    Returns:
        Track information dictionary or None if not found
    """
    cache_key = f"track:info:{track_id}"
    
    # Check cache
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        track = sp.track(track_id)
        
        if not track:
            return None
        
        # Extract relevant information
        info = {
            'id': track['id'],
            'name': track['name'],
            'artists': [artist['name'] for artist in track['artists']],
            'artist_ids': [artist['id'] for artist in track['artists']],
            'album': track['album']['name'],
            'album_id': track['album']['id'],
            'duration_ms': track['duration_ms'],
            'explicit': track.get('explicit', False),
            'popularity': track.get('popularity', 0),
            'preview_url': track.get('preview_url'),
            'external_url': track['external_urls'].get('spotify'),
            'uri': track['uri'],
        }
        
        # Add album art if available
        if track['album'].get('images') and len(track['album']['images']) > 0:
            info['album_art'] = track['album']['images'][0]['url']
        
        # Cache for 1 day
        await cache_service.set_in_cache(cache_key, info, expiration=86400)
        
        return info
        
    except spotipy.SpotifyException as e:
        print(f"⚠️ Spotify API error getting track {track_id}: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Error getting track info: {e}")
        return None


async def get_artist_info(
    artist_id: str,
    access_token: Optional[str] = None
) -> Optional[Dict]:
    """
    Get artist information including genres.
    
    Args:
        artist_id: Spotify artist ID
        access_token: User's access token (optional)
        
    Returns:
        Artist information dictionary or None if not found
    """
    cache_key = f"artist:info:{artist_id}"
    
    # Check cache
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        artist = sp.artist(artist_id)
        
        if not artist:
            return None
        
        # Extract relevant information
        info = {
            'id': artist['id'],
            'name': artist['name'],
            'genres': artist.get('genres', []),
            'popularity': artist.get('popularity', 0),
            'followers': artist['followers'].get('total', 0),
            'external_url': artist['external_urls'].get('spotify'),
        }
        
        # Add artist image if available
        if artist.get('images') and len(artist['images']) > 0:
            info['image_url'] = artist['images'][0]['url']
        
        # Cache for 1 week (artist info changes slowly)
        await cache_service.set_in_cache(cache_key, info, expiration=604800)
        
        return info
        
    except spotipy.SpotifyException as e:
        print(f"⚠️ Spotify API error getting artist {artist_id}: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Error getting artist info: {e}")
        return None


async def get_tracks_with_features(
    track_ids: List[str],
    access_token: Optional[str] = None
) -> List[Dict]:
    """
    Get complete track information including audio features.
    
    Args:
        track_ids: List of Spotify track IDs
        access_token: User's access token (optional)
        
    Returns:
        List of dictionaries with track info and audio features
    """
    if not track_ids:
        return []
    
    print(f"🎵 Fetching complete data for {len(track_ids)} tracks...")
    
    # Get audio features (with caching)
    audio_features = await get_audio_features(track_ids, access_token)
    
    # Get track information
    tracks_data = []
    
    try:
        sp = get_spotify_client(access_token)
        
        # Fetch in batches of 50 (Spotify limit)
        for i in range(0, len(track_ids), 50):
            batch_ids = track_ids[i:i+50]
            
            try:
                tracks_response = sp.tracks(batch_ids)
                
                for j, track in enumerate(tracks_response['tracks']):
                    if not track:
                        continue
                    
                    track_idx = i + j
                    features = audio_features[track_idx] if track_idx < len(audio_features) else None
                    
                    track_data = {
                        'id': track['id'],
                        'name': track['name'],
                        'artists': [artist['name'] for artist in track['artists']],
                        'artist_ids': [artist['id'] for artist in track['artists']],
                        'album': track['album']['name'],
                        'duration_ms': track['duration_ms'],
                        'popularity': track.get('popularity', 0),
                        'preview_url': track.get('preview_url'),
                        'external_url': track['external_urls'].get('spotify'),
                        'features': features
                    }
                    
                    tracks_data.append(track_data)
                    
            except spotipy.SpotifyException as e:
                print(f"⚠️ Spotify API error in batch: {e}")
                continue
        
        print(f"✅ Retrieved complete data for {len(tracks_data)} tracks")
        
    except Exception as e:
        print(f"⚠️ Error getting tracks with features: {e}")
    
    return tracks_data


async def search_tracks(
    query: str,
    limit: int = 20,
    access_token: Optional[str] = None
) -> List[Dict]:
    """
    Search for tracks on Spotify.
    
    Args:
        query: Search query
        limit: Maximum number of results
        access_token: User's access token (optional)
        
    Returns:
        List of track information dictionaries
    """
    try:
        sp = get_spotify_client(access_token)
        results = sp.search(q=query, type='track', limit=limit)
        
        tracks = []
        for item in results['tracks']['items']:
            tracks.append({
                'id': item['id'],
                'name': item['name'],
                'artists': [artist['name'] for artist in item['artists']],
                'album': item['album']['name'],
                'popularity': item.get('popularity', 0),
                'preview_url': item.get('preview_url'),
                'external_url': item['external_urls'].get('spotify'),
            })
        
        return tracks
        
    except spotipy.SpotifyException as e:
        print(f"⚠️ Spotify API error searching: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Error searching tracks: {e}")
        return []


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
    
    Args:
        seed_tracks: Up to 5 track IDs
        seed_artists: Up to 5 artist IDs
        seed_genres: Up to 5 genre names
        target_valence: Target valence (0-1)
        target_energy: Target energy (0-1)
        limit: Number of recommendations
        access_token: User's access token (optional)
        
    Returns:
        List of recommended tracks
    """
    try:
        sp = get_spotify_client(access_token)
        
        # Ensure we don't exceed Spotify's limit of 5 seeds total
        seeds = {
            'seed_tracks': (seed_tracks or [])[:5],
            'seed_artists': (seed_artists or [])[:5],
            'seed_genres': (seed_genres or [])[:5]
        }
        
        # Remove empty seeds
        seeds = {k: v for k, v in seeds.items() if v}
        
        # Add target features if specified
        kwargs = {}
        if target_valence is not None:
            kwargs['target_valence'] = target_valence
        if target_energy is not None:
            kwargs['target_energy'] = target_energy
        
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
            })
        
        print(f"✅ Got {len(recommendations)} recommendations")
        return recommendations
        
    except spotipy.SpotifyException as e:
        print(f"⚠️ Spotify API error getting recommendations: {e}")
        return []
    except Exception as e:
        print(f"⚠️ Error getting recommendations: {e}")
        return []


async def get_available_genre_seeds(
    access_token: Optional[str] = None
) -> List[str]:
    """
    Get available genre seeds for recommendations.
    
    Args:
        access_token: User's access token (optional)
        
    Returns:
        List of genre names
    """
    cache_key = "spotify:genre_seeds"
    
    # Check cache (genres don't change often)
    cached = await cache_service.get_from_cache(cache_key)
    if cached:
        return cached
    
    try:
        sp = get_spotify_client(access_token)
        genres = sp.recommendation_genre_seeds()
        
        genre_list = genres.get('genres', [])
        
        # Cache for 1 week
        await cache_service.set_in_cache(cache_key, genre_list, expiration=604800)
        
        return genre_list
        
    except Exception as e:
        print(f"⚠️ Error getting genre seeds: {e}")
        return []


# Utility functions for testing

async def test_spotify_service():
    """
    Test the Spotify service with sample operations.
    """
    print("\n" + "="*60)
    print("🧪 Testing Spotify Service")
    print("="*60)
    
    # Test track IDs (popular songs)
    test_track_ids = [
        "3n3Ppam7vgaVa1iaRUc9Lp",  # Mr. Brightside - The Killers
        "0VjIjW4GlUZAMYd2vXMi3b",  # Blinding Lights - The Weeknd
    ]
    
    print("\n1. Testing audio features...")
    features = await get_audio_features(test_track_ids)
    print(f"   Retrieved {len(features)} feature sets")
    
    if features and features[0]:
        print(f"   Sample features:")
        print(f"   Valence: {features[0]['valence']:.3f}")
        print(f"   Energy: {features[0]['energy']:.3f}")
        print(f"   Tempo: {features[0]['tempo']:.1f} BPM")
    
    print("\n2. Testing track info...")
    if test_track_ids:
        track_info = await get_track_info(test_track_ids[0])
        if track_info:
            print(f"   Track: {track_info['name']}")
            print(f"   Artist: {', '.join(track_info['artists'])}")
            print(f"   Popularity: {track_info['popularity']}")
    
    print("\n3. Testing search...")
    search_results = await search_tracks("happy", limit=5)
    print(f"   Found {len(search_results)} tracks")
    for track in search_results[:3]:
        print(f"   - {track['name']} by {', '.join(track['artists'])}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_spotify_service())