"""
Lyrics fetching and sentiment analysis service for Moodify-AI.

Features:
- Fetch lyrics from Genius API
- Multi-language support with auto-detection
- Sentiment analysis using TextBlob
- Translation for non-English lyrics
- Caching for performance
"""

import os
from typing import Dict, Optional
import lyricsgenius
from textblob import TextBlob
from langdetect import detect, LangDetectException
from . import cache_service


# Global Genius client
genius: Optional[lyricsgenius.Genius] = None


def get_genius_client() -> lyricsgenius.Genius:
    """
    Initialize and return Genius API client.
    
    Returns:
        Configured Genius client
    """
    global genius
    
    if genius is None:
        api_key = os.getenv("GENIUS_API_KEY")
        
        if not api_key:
            print("⚠️ GENIUS_API_KEY not found in environment")
            print("   Lyrics sentiment analysis will return neutral values")
            return None
        
        try:
            genius = lyricsgenius.Genius(
                api_key,
                skip_non_songs=True,
                excluded_terms=["(Remix)", "(Live)"],
                remove_section_headers=True,
                timeout=10,
                retries=2
            )
            
            # Reduce verbosity
            genius.verbose = False
            
            print("✅ Genius API client initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize Genius client: {e}")
            genius = None
    
    return genius


def clean_lyrics(lyrics: str) -> str:
    """
    Clean and normalize lyrics text.
    
    Args:
        lyrics: Raw lyrics string
        
    Returns:
        Cleaned lyrics
    """
    if not lyrics:
        return ""
    
    # Remove common artifacts
    lyrics = lyrics.replace('\n', ' ')
    lyrics = lyrics.replace('  ', ' ')
    
    # Remove Genius annotations (text in brackets)
    import re
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
    # Remove "Embed" text that sometimes appears
    lyrics = lyrics.replace('Embed', '')
    
    # Remove excessive whitespace
    lyrics = ' '.join(lyrics.split())
    
    return lyrics.strip()


def detect_language(text: str) -> str:
    """
    Detect the language of text.
    
    Args:
        text: Text to analyze
        
    Returns:
        ISO language code (e.g., 'en', 'es', 'fr')
    """
    if not text or len(text.strip()) < 10:
        return 'en'  # Default to English for short text
    
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        print("⚠️ Language detection failed, defaulting to English")
        return 'en'


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text to English for sentiment analysis.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        
    Returns:
        Translated text (or original if translation fails)
    """
    if source_lang == 'en':
        return text
    
    try:
        from deep_translator import GoogleTranslator
        
        # Split into chunks if too long (Google Translate has 5000 char limit)
        max_length = 4500  # Leave some buffer
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            translated_chunks = []
            
            for chunk in chunks:
                translated = GoogleTranslator(source=source_lang, target='en').translate(chunk)
                translated_chunks.append(translated)
            
            return ' '.join(translated_chunks)
        else:
            translated = GoogleTranslator(source=source_lang, target='en').translate(text)
            return translated
            
    except Exception as e:
        print(f"⚠️ Translation failed ({source_lang} → en): {e}")
        return text  # Return original if translation fails


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with polarity and subjectivity scores
    """
    if not text or len(text.strip()) < 5:
        return {
            "polarity": 0.0,
            "subjectivity": 0.0
        }
    
    try:
        blob = TextBlob(text)
        
        return {
            "polarity": blob.sentiment.polarity,      # -1 (negative) to 1 (positive)
            "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        }
    except Exception as e:
        print(f"⚠️ Sentiment analysis failed: {e}")
        return {
            "polarity": 0.0,
            "subjectivity": 0.0
        }


async def get_lyrics_sentiment(
    track_name: str, 
    artist_name: str,
    enable_translation: bool = True
) -> Dict[str, float]:
    """
    Fetch lyrics and analyze sentiment with multi-language support.
    
    Args:
        track_name: Name of the track
        artist_name: Name of the artist
        enable_translation: Whether to translate non-English lyrics
        
    Returns:
        Dictionary with sentiment scores and metadata
    """
    # Check cache first
    cached = await cache_service.get_cached_lyrics_sentiment(track_name, artist_name)
    if cached:
        print(f"📦 Cache HIT: Lyrics sentiment for {track_name}")
        return cached
    
    print(f"🔍 Fetching lyrics for: {track_name} by {artist_name}")
    
    # Get Genius client
    genius_client = get_genius_client()
    
    if not genius_client:
        # Return neutral sentiment if Genius API not available
        neutral_sentiment = {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "language": "unknown",
            "translated": False,
            "lyrics_found": False
        }
        # Cache neutral result for a shorter time
        await cache_service.cache_lyrics_sentiment(
            track_name, 
            artist_name, 
            neutral_sentiment
        )
        return neutral_sentiment
    
    try:
        # Search for song on Genius
        song = genius_client.search_song(track_name, artist_name)
        
        if not song or not song.lyrics:
            print(f"⚠️ Lyrics not found for: {track_name}")
            
            neutral_sentiment = {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "language": "unknown",
                "translated": False,
                "lyrics_found": False
            }
            
            # Cache for 1 week (lyrics won't change)
            await cache_service.cache_lyrics_sentiment(
                track_name,
                artist_name,
                neutral_sentiment
            )
            
            return neutral_sentiment
        
        # Clean lyrics
        lyrics = clean_lyrics(song.lyrics)
        
        if not lyrics or len(lyrics) < 20:
            print(f"⚠️ Lyrics too short or empty for: {track_name}")
            neutral_sentiment = {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "language": "unknown",
                "translated": False,
                "lyrics_found": False
            }
            await cache_service.cache_lyrics_sentiment(
                track_name,
                artist_name,
                neutral_sentiment
            )
            return neutral_sentiment
        
        print(f"✅ Lyrics found ({len(lyrics)} chars)")
        
        # Detect language
        detected_lang = detect_language(lyrics)
        print(f"🌍 Detected language: {detected_lang}")
        
        # Translate if not English and translation is enabled
        translated = False
        analysis_text = lyrics
        
        if enable_translation and detected_lang != 'en':
            print(f"🔄 Translating from {detected_lang} to English...")
            translated_text = translate_to_english(lyrics, detected_lang)
            
            if translated_text != lyrics:
                analysis_text = translated_text
                translated = True
                print(f"✅ Translation complete")
        
        # Analyze sentiment
        sentiment = analyze_sentiment(analysis_text)
        
        # Add metadata
        result = {
            "polarity": sentiment["polarity"],
            "subjectivity": sentiment["subjectivity"],
            "language": detected_lang,
            "translated": translated,
            "lyrics_found": True,
            "lyrics_length": len(lyrics)
        }
        
        print(f"📊 Sentiment: polarity={sentiment['polarity']:.2f}, "
              f"subjectivity={sentiment['subjectivity']:.2f}")
        
        # Cache result for 1 week
        await cache_service.cache_lyrics_sentiment(
            track_name,
            artist_name,
            result
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Error fetching lyrics for {track_name}: {e}")
        
        # Return neutral sentiment on error
        neutral_sentiment = {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "language": "unknown",
            "translated": False,
            "lyrics_found": False,
            "error": str(e)
        }
        
        # Cache error result for 1 hour (might be temporary)
        await cache_service.set_in_cache(
            f"lyrics:{track_name}:{artist_name}",
            neutral_sentiment,
            expiration=3600
        )
        
        return neutral_sentiment


async def batch_get_lyrics_sentiment(
    tracks: list,
    max_concurrent: int = 5
) -> Dict[str, Dict]:
    """
    Fetch lyrics sentiment for multiple tracks concurrently.
    
    Args:
        tracks: List of dicts with 'name' and 'artist' keys
        max_concurrent: Maximum concurrent requests
        
    Returns:
        Dictionary mapping track keys to sentiment data
    """
    import asyncio
    
    results = {}
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_semaphore(track):
        async with semaphore:
            track_key = f"{track['name']}_{track['artist']}"
            sentiment = await get_lyrics_sentiment(
                track['name'],
                track['artist']
            )
            return track_key, sentiment
    
    # Create tasks
    tasks = [fetch_with_semaphore(track) for track in tracks]
    
    # Execute concurrently
    print(f"🎵 Fetching lyrics for {len(tracks)} tracks (max {max_concurrent} concurrent)...")
    completed = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in completed:
        if isinstance(result, Exception):
            print(f"❌ Error in batch fetch: {result}")
            continue
        
        track_key, sentiment = result
        results[track_key] = sentiment
    
    print(f"✅ Completed batch lyrics fetch: {len(results)}/{len(tracks)} successful")
    
    return results


def get_mood_from_lyrics(sentiment: Dict) -> str:
    """
    Determine mood category from lyrics sentiment.
    
    Args:
        sentiment: Sentiment dictionary with polarity and subjectivity
        
    Returns:
        Mood string: 'Positive', 'Negative', or 'Neutral'
    """
    polarity = sentiment.get('polarity', 0.0)
    subjectivity = sentiment.get('subjectivity', 0.0)
    
    # High subjectivity = emotional lyrics
    if subjectivity > 0.5:
        if polarity > 0.2:
            return "Positive"
        elif polarity < -0.2:
            return "Negative"
        else:
            return "Mixed"
    else:
        # Low subjectivity = more objective/factual
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"


def calculate_lyrics_weight(sentiment: Dict) -> float:
    """
    Calculate how much weight lyrics should have in mood prediction.
    
    Strongly emotional lyrics get more weight.
    
    Args:
        sentiment: Sentiment dictionary
        
    Returns:
        Weight value between 0.0 and 1.0
    """
    polarity = abs(sentiment.get('polarity', 0.0))
    subjectivity = sentiment.get('subjectivity', 0.0)
    
    # Strong sentiment + high subjectivity = high weight
    strength = (polarity * 0.7) + (subjectivity * 0.3)
    
    # Scale to 0.2-0.8 range
    weight = 0.2 + (strength * 0.6)
    
    return min(weight, 0.8)


# Utility functions for testing

async def test_lyrics_service():
    """
    Test the lyrics service with sample tracks.
    """
    print("\n" + "="*60)
    print("🧪 Testing Lyrics Service")
    print("="*60)
    
    test_tracks = [
        {"name": "Happy", "artist": "Pharrell Williams"},
        {"name": "Someone Like You", "artist": "Adele"},
        {"name": "Stairway to Heaven", "artist": "Led Zeppelin"},
    ]
    
    for track in test_tracks:
        print(f"\n📝 Testing: {track['name']} by {track['artist']}")
        sentiment = await get_lyrics_sentiment(track['name'], track['artist'])
        
        print(f"   Polarity: {sentiment['polarity']:.3f}")
        print(f"   Subjectivity: {sentiment['subjectivity']:.3f}")
        print(f"   Language: {sentiment.get('language', 'unknown')}")
        print(f"   Translated: {sentiment.get('translated', False)}")
        print(f"   Lyrics found: {sentiment.get('lyrics_found', False)}")
        print(f"   Mood: {get_mood_from_lyrics(sentiment)}")
        print(f"   Weight: {calculate_lyrics_weight(sentiment):.3f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_lyrics_service())