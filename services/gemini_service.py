"""
Gemini AI Service for Audio Feature Estimation
Provides fallback feature extraction when MusicBrainz/AcousticBrainz data unavailable

Usage:
    from services import gemini_service
    
    features = await gemini_service.estimate_audio_features_with_gemini(
        "Blinding Lights",
        "The Weeknd"
    )
"""

import os
import json
from typing import Dict, Optional, List

# 🔄 New official SDK (install: pip install google-genai)
from google import genai

from . import cache_service


# Gemini Configuration - Delay initialization
# Initialize these at module level
GEMINI_API_KEY = None
_initialized = False

# New client handle (google-genai)
_client: Optional[genai.Client] = None

# Default model preference (as of Nov 2025)
# We'll also fallback to 1.5 if 2.5 isn't accessible for the key
PRIMARY_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL_NAMES = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
]


def _init_gemini():
    """Initialize Gemini API (called lazily on first use)"""
    global GEMINI_API_KEY, _initialized, _client
    
    if _initialized:
        return
    
    _initialized = True
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if GEMINI_API_KEY:
        try:
            # New client constructor (google-genai)
            _client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"✅ Gemini AI initialized (key: {GEMINI_API_KEY[:8]}...)")
        except Exception as e:
            print(f"⚠️  Gemini AI initialization failed: {e}")
            GEMINI_API_KEY = None
            _client = None
    else:
        print("⚠️  GEMINI_API_KEY not found in environment")
        print("   Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file to enable AI-powered feature estimation")


def _extract_text_from_response(response) -> str:
    """
    Robustly extract plain text from google-genai responses.
    Different SDK versions may expose .text or .output_text, or candidates.
    """
    if response is None:
        return ""
    # Preferred attributes
    for attr in ("text", "output_text"):
        if hasattr(response, attr) and getattr(response, attr):
            return getattr(response, attr)
    # Try candidates
    try:
        cands = getattr(response, "candidates", None)
        if cands:
            for c in cands:
                # candidate.content.parts may exist
                content = getattr(c, "content", None)
                if content and hasattr(content, "parts"):
                    parts = content.parts or []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            return t
                # or candidate.text directly
                t = getattr(c, "text", None)
                if t:
                    return t
    except Exception:
        pass
    # Try to stringify as a last resort
    try:
        return str(response)
    except Exception:
        return ""


def _call_gemini_generate(prompt: str) -> Optional[str]:
    """
    Call Gemini with primary model and fallbacks, return response text or None on failure.
    Uses google-genai client.
    """
    global _client
    if not _client:
        return None
    
    model_chain = [PRIMARY_MODEL_NAME] + [m for m in FALLBACK_MODEL_NAMES if m != PRIMARY_MODEL_NAME]
    
    last_error = None
    for model_name in model_chain:
        try:
            # google-genai call pattern
            response = _client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            text = _extract_text_from_response(response)
            if text and isinstance(text, str) and text.strip():
                if model_name != PRIMARY_MODEL_NAME:
                    print(f"ℹ️ Used fallback model: {model_name}")
                return text
            else:
                print(f"⚠️ Empty response text from model '{model_name}'")
        except Exception as e:
            last_error = e
            print(f"⚠️ Gemini generate_content error with '{model_name}': {e}")
            continue
    
    if last_error:
        print(f"❌ Gemini generate_content failed on all models. Last error: {last_error}")
    return None


async def estimate_audio_features_with_gemini(
    track_name: str,
    artist_name: str,
    album_name: Optional[str] = None,
    genre: Optional[str] = None,
    lyrics_snippet: Optional[str] = None
) -> Optional[Dict]:
    """
    Use Gemini AI to estimate audio features when MBID/AcousticBrainz unavailable
    
    Args:
        track_name: Track name
        artist_name: Artist name
        album_name: Album name (optional, helps with context)
        genre: Genre (optional, helps with estimation)
        lyrics_snippet: First few lines of lyrics (optional, improves accuracy)
        
    Returns:
        Estimated audio features dictionary in Spotify-like format
    """
    # Initialize Gemini on first use
    _init_gemini()
    
    # Check cache first
    cache_key = f"gemini:features:{track_name}:{artist_name}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 Cache HIT: Gemini features")
        return cached
    
    if not GEMINI_API_KEY or not _client:
        print("⚠️ Gemini API key not configured")
        return None
    
    try:
        print(f"🤖 Estimating features with Gemini AI: {track_name} by {artist_name}")
        
        # Build context-rich prompt
        prompt = build_feature_estimation_prompt(
            track_name,
            artist_name,
            album_name,
            genre,
            lyrics_snippet
        )
        
        # Call Gemini API (new client syntax with fallback)
        response_text = _call_gemini_generate(prompt)
        if not response_text:
            print("⚠️ Gemini returned no usable text")
            return None
        
        # Parse response
        features = parse_gemini_response(response_text)
        
        if features:
            print(f"✅ Gemini estimated features successfully")
            print(f"   Valence: {features['valence']:.3f}, Energy: {features['energy']:.3f}")
            
            # Cache for 6 hours (temporary storage)
            await cache_service.set_in_cache(cache_key, features, expiration=21600)
            
            return features
        else:
            print("⚠️ Failed to parse Gemini response")
            return None
            
    except Exception as e:
        print(f"❌ Gemini feature estimation error: {e}")
        return None


def build_feature_estimation_prompt(
    track_name: str,
    artist_name: str,
    album_name: Optional[str] = None,
    genre: Optional[str] = None,
    lyrics_snippet: Optional[str] = None
) -> str:
    """
    Build a detailed prompt for Gemini to estimate audio features
    """
    prompt = f"""You are an expert music analyst. Analyze the song and estimate its audio characteristics.

Song Information:
- Title: "{track_name}"
- Artist: {artist_name}"""
    
    if album_name:
        prompt += f"\n- Album: {album_name}"
    
    if genre:
        prompt += f"\n- Genre: {genre}"
    
    if lyrics_snippet:
        prompt += f"\n- Lyrics excerpt: \"{lyrics_snippet[:200]}...\""
    
    prompt += """

Based on your knowledge of this song (or similar songs by this artist/genre), estimate the following audio features:

1. **valence** (0.0 to 1.0): Musical positiveness/happiness
   - 0.0 = Very sad, depressed, angry
   - 0.5 = Neutral
   - 1.0 = Very happy, cheerful, euphoric

2. **energy** (0.0 to 1.0): Intensity and activity level
   - 0.0 = Calm, quiet, ambient
   - 0.5 = Moderate
   - 1.0 = Fast, loud, energetic, intense

3. **danceability** (0.0 to 1.0): How suitable for dancing
   - Consider tempo, rhythm stability, beat strength
   - 0.0 = Not danceable
   - 1.0 = Very danceable

4. **acousticness** (0.0 to 1.0): Confidence of acoustic vs. electric
   - 0.0 = Heavily electronic/processed
   - 1.0 = Purely acoustic instruments

5. **instrumentalness** (0.0 to 1.0): Presence of vocals
   - 0.0 = Lots of vocals
   - 1.0 = No vocals/instrumental

6. **speechiness** (0.0 to 1.0): Presence of spoken words
   - 0.0 = Music only
   - 0.3-0.6 = Music with some rap/speech
   - 1.0 = Audiobook/poetry

7. **tempo** (BPM): Estimated beats per minute (40-200 typical range)

8. **loudness** (dB): Average loudness in decibels (-60 to 0)
   - Loud rock/EDM: -5 to -3
   - Normal: -10 to -6
   - Quiet/acoustic: -15 to -11

9. **liveness** (0.0 to 1.0): Presence of audience/live recording
   - 0.0 = Studio recording
   - 0.8+ = Live performance

10. **tags**: 3-5 mood/genre tags (e.g., ["upbeat", "pop", "summer", "party"])

Return ONLY valid JSON in this exact format (no markdown, no explanations):
{
  "valence": 0.75,
  "energy": 0.80,
  "danceability": 0.85,
  "acousticness": 0.10,
  "instrumentalness": 0.05,
  "speechiness": 0.15,
  "tempo": 128.0,
  "loudness": -5.5,
  "liveness": 0.12,
  "tags": ["happy", "pop", "dance", "energetic"]
}"""
    
    return prompt


def parse_gemini_response(response_text: str) -> Optional[Dict]:
    """
    Parse Gemini's JSON response and validate features
    """
    try:
        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        data = json.loads(response_text)
        
        # Validate and normalize features
        features = {
            'id': 'gemini_estimated',
            'valence': float(data.get('valence', 0.5)),
            'energy': float(data.get('energy', 0.5)),
            'danceability': float(data.get('danceability', 0.5)),
            'acousticness': float(data.get('acousticness', 0.5)),
            'instrumentalness': float(data.get('instrumentalness', 0.3)),
            'speechiness': float(data.get('speechiness', 0.1)),
            'tempo': float(data.get('tempo', 120.0)),
            'loudness': float(data.get('loudness', -10.0)),
            'liveness': float(data.get('liveness', 0.1)),
            'key': 0,  # Can't estimate accurately
            'mode': 1,  # Default to major
            'time_signature': 4,  # Default to 4/4
            'duration_ms': 0,  # Unknown
            'tags': data.get('tags', []),
            'source': 'gemini_ai'
        }
        
        # Clamp values to valid ranges
        for key in ['valence', 'energy', 'danceability', 'acousticness', 
                    'instrumentalness', 'speechiness', 'liveness']:
            features[key] = max(0.0, min(1.0, features[key]))
        
        features['tempo'] = max(40.0, min(200.0, features['tempo']))
        features['loudness'] = max(-60.0, min(0.0, features['loudness']))
        
        return features
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse Gemini JSON response: {e}")
        print(f"   Response: {response_text[:200]}")
        return None
        
    except Exception as e:
        print(f"❌ Error parsing Gemini response: {e}")
        return None


async def batch_estimate_features(
    tracks: List[Dict],
    max_concurrent: int = 3
) -> List[Optional[Dict]]:
    """
    Estimate features for multiple tracks with Gemini
    Rate-limited to avoid API throttling
    
    Args:
        tracks: List of dicts with 'name' and 'artist' keys
        max_concurrent: Maximum concurrent API calls
        
    Returns:
        List of feature dictionaries
    """
    import asyncio
    
    # Initialize Gemini on first use
    _init_gemini()
    
    print(f"🤖 Batch estimating features for {len(tracks)} tracks with Gemini")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def estimate_with_limit(track):
        async with semaphore:
            return await estimate_audio_features_with_gemini(
                track.get('name'),
                track.get('artist'),
                track.get('album'),
                track.get('genre')
            )
    
    tasks = [estimate_with_limit(track) for track in tracks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"⚠️ Error estimating features for track {i}: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)
    
    successful = len([r for r in processed_results if r])
    print(f"✅ Gemini batch estimation complete: {successful}/{len(tracks)} successful")
    
    return processed_results


async def get_tags_with_gemini(
    track_name: str,
    artist_name: str
) -> List[str]:
    """
    Get genre/mood tags using Gemini as fallback for Last.fm
    
    Args:
        track_name: Track name
        artist_name: Artist name
        
    Returns:
        List of tags
    """
    # Initialize Gemini on first use
    _init_gemini()
    
    cache_key = f"gemini:tags:{track_name}:{artist_name}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached
    
    if not GEMINI_API_KEY or not _client:
        return []
    
    try:
        prompt = f"""List 5-8 genre and mood tags for the song "{track_name}" by {artist_name}.

Tags should be lowercase, single words or short phrases like:
- Genre tags: pop, rock, hip-hop, jazz, electronic, etc.
- Mood tags: happy, sad, energetic, calm, aggressive, romantic, etc.
- Style tags: upbeat, mellow, dark, bright, etc.

Return ONLY a JSON array of strings:
["tag1", "tag2", "tag3", ...]"""
        
        response_text = _call_gemini_generate(prompt)
        if not response_text:
            return []
        
        # Parse response
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:-3]
        
        try:
            tags = json.loads(response_text)
        except Exception:
            # If model returned plain text list (comma separated), normalize
            parts = [p.strip() for p in response_text.split(",") if p.strip()]
            tags = parts
        
        if isinstance(tags, list):
            tags = [tag.lower().strip() for tag in tags if isinstance(tag, str)]
            print(f"✅ Gemini generated {len(tags)} tags")
            
            # Cache for 1 week
            await cache_service.set_in_cache(cache_key, tags, expiration=604800)
            return tags
        
        return []
        
    except Exception as e:
        print(f"❌ Gemini tags error: {e}")
        return []


async def cleanup_gemini_cache():
    """
    Clean up temporary Gemini-generated features from cache
    Useful for keeping cache size manageable
    """
    try:
        deleted = await cache_service.delete_pattern("gemini:*")
        print(f"🧹 Cleaned up {deleted} Gemini cache entries")
        return deleted
    except Exception as e:
        print(f"⚠️ Error cleaning up Gemini cache: {e}")
        return 0


# Testing function
async def test_gemini_service():
    """Test Gemini feature estimation"""
    print("\n" + "="*60)
    print("🧪 Testing Gemini AI Feature Estimation")
    print("="*60)
    
    test_tracks = [
        {"name": "Shape of You", "artist": "Ed Sheeran", "genre": "pop"},
        {"name": "Bohemian Rhapsody", "artist": "Queen", "genre": "rock"},
        {"name": "Lose Yourself", "artist": "Eminem", "genre": "hip-hop"}
    ]
    
    for track in test_tracks:
        print(f"\n🔍 Testing: {track['name']} by {track['artist']}")
        
        features = await estimate_audio_features_with_gemini(
            track['name'],
            track['artist'],
            genre=track.get('genre')
        )
        
        if features:
            print(f"   ✅ Valence: {features['valence']:.3f}")
            print(f"   ✅ Energy: {features['energy']:.3f}")
            print(f"   ✅ Tempo: {features['tempo']:.1f} BPM")
            print(f"   ✅ Tags: {', '.join(features.get('tags', []))}")
        else:
            print(f"   ❌ Failed to estimate features")
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_gemini_service())
