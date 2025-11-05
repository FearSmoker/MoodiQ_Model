"""
Lyrics fetching and sentiment analysis service for MoodiQ-AI.

Features:
- Fetch lyrics from Genius API (direct REST call)
- Multi-language support with auto-detection
- Sentiment analysis using TextBlob
- Translation for non-English lyrics
- Gemini AI fallback (only in worst-case)
- Caching for performance
"""

import os
import re
import html
import requests
from typing import Dict, Optional
from textblob import TextBlob
from langdetect import detect, LangDetectException
from . import cache_service
from . import gemini_service  # ✅ Gemini fallback integration


# ============================================================
# Genius API: Direct REST client replacement (no lyricsgenius)
# ============================================================

def fetch_lyrics_from_genius(track_name: str, artist_name: str) -> Optional[str]:
    """
    Fetch lyrics from Genius API via REST + HTML scraping.
    """
    access_token = os.getenv("GENIUS_API_KEY")
    if not access_token:
        print("⚠️ GENIUS_API_KEY missing")
        return None

    headers = {"Authorization": f"Bearer {access_token}"}
    query = f"{track_name} {artist_name}"

    try:
        # Step 1: Search for song
        response = requests.get(
            "https://api.genius.com/search",
            headers=headers,
            params={"q": query},
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        hits = data.get("response", {}).get("hits", [])
        if not hits:
            print(f"⚠️ No Genius results for {query}")
            return None

        # Step 2: Get the song URL from top hit
        song_url = hits[0]["result"]["url"]

        # Step 3: Scrape lyrics from Genius webpage
        page = requests.get(song_url, timeout=10)
        lyrics = re.findall(r'<div[^>]*class="Lyrics__Container[^>]*>(.*?)</div>', page.text, re.S)
        if lyrics:
            text = " ".join(html.unescape(re.sub(r"<.*?>", "", l)) for l in lyrics)
            return text.strip()
        else:
            print(f"⚠️ Could not extract lyrics for {query}")
            return None

    except Exception as e:
        print(f"❌ Genius fetch failed for {query}: {e}")
        return None


# Dummy placeholder (compatibility)
genius = True
def get_genius_client():
    """Dummy Genius client (kept for backward compatibility)."""
    return genius


# ============================================================
# Utility functions: cleaning, language detection, translation
# ============================================================

def clean_lyrics(lyrics: str) -> str:
    """
    Clean and normalize lyrics text.
    """
    if not lyrics:
        return ""
    lyrics = lyrics.replace('\n', ' ').replace('  ', ' ')
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    lyrics = lyrics.replace('Embed', '')
    lyrics = ' '.join(lyrics.split())
    return lyrics.strip()


def detect_language(text: str) -> str:
    """
    Detect the language of text.
    """
    if not text or len(text.strip()) < 10:
        return 'en'
    try:
        return detect(text)
    except LangDetectException:
        print("⚠️ Language detection failed, defaulting to English")
        return 'en'


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text to English for sentiment analysis.
    """
    if source_lang == 'en':
        return text
    try:
        from deep_translator import GoogleTranslator
        max_length = 4500
        if len(text) > max_length:
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            translated_chunks = [
                GoogleTranslator(source=source_lang, target='en').translate(chunk)
                for chunk in chunks
            ]
            return ' '.join(translated_chunks)
        else:
            return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception as e:
        print(f"⚠️ Translation failed ({source_lang} → en): {e}")
        return text


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using TextBlob.
    """
    if not text or len(text.strip()) < 5:
        return {"polarity": 0.0, "subjectivity": 0.0}
    try:
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    except Exception as e:
        print(f"⚠️ Sentiment analysis failed: {e}")
        return {"polarity": 0.0, "subjectivity": 0.0}


# ============================================================
# Main: Lyrics fetching + sentiment analysis + caching
# ============================================================

async def get_lyrics_sentiment(
    track_name: str,
    artist_name: str,
    enable_translation: bool = True
) -> Dict[str, float]:
    """
    Fetch lyrics and analyze sentiment with multi-language support.
    Falls back to Gemini AI estimation only in the worst case.
    """
    # 1️⃣ Cache check
    cached = await cache_service.get_cached_lyrics_sentiment(track_name, artist_name)
    if cached:
        print(f"📦 Cache HIT: Lyrics sentiment for {track_name}")
        return cached

    print(f"🔍 Fetching lyrics for: {track_name} by {artist_name}")
    genius_client = get_genius_client()

    if not genius_client:
        print("⚠️ Genius client unavailable")
        return await _handle_lyrics_failure(track_name, artist_name)

    try:
        # 2️⃣ Fetch lyrics (direct REST)
        lyrics = fetch_lyrics_from_genius(track_name, artist_name)
        if not lyrics:
            print(f"⚠️ Lyrics not found for: {track_name}")
            return await _handle_lyrics_failure(track_name, artist_name, track_name, artist_name)

        # 3️⃣ Clean lyrics
        lyrics = clean_lyrics(lyrics)
        if not lyrics or len(lyrics) < 20:
            print(f"⚠️ Lyrics too short or empty for: {track_name}")
            return await _handle_lyrics_failure(track_name, artist_name, track_name, artist_name)

        print(f"✅ Lyrics found ({len(lyrics)} chars)")

        # 4️⃣ Detect language
        detected_lang = detect_language(lyrics)
        print(f"🌍 Detected language: {detected_lang}")

        translated = False
        analysis_text = lyrics

        # 5️⃣ Translate if needed
        if enable_translation and detected_lang != 'en':
            print(f"🔄 Translating from {detected_lang} to English...")
            translated_text = translate_to_english(lyrics, detected_lang)
            if translated_text != lyrics:
                analysis_text = translated_text
                translated = True
                print(f"✅ Translation complete")

        # 6️⃣ Sentiment analysis
        sentiment = analyze_sentiment(analysis_text)
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

        # 7️⃣ Cache for 1 week
        await cache_service.cache_lyrics_sentiment(track_name, artist_name, result)
        return result

    except Exception as e:
        print(f"❌ Error fetching lyrics for {track_name}: {e}")
        return await _handle_lyrics_failure(track_name, artist_name, track_name, artist_name, error=str(e))


# ============================================================
# Fallback handler with Gemini AI
# ============================================================

async def _handle_lyrics_failure(track_name, artist_name, album_name=None, genre=None, error: Optional[str] = None):
    """
    Handle cases where lyrics or sentiment fail.
    Tries Gemini AI as the final fallback (only if everything else fails).
    """
    try:
        print(f"🧠 Using Gemini AI as last-resort fallback for {track_name} by {artist_name}")
        features = await gemini_service.estimate_audio_features_with_gemini(
            track_name, artist_name, album_name=album_name, genre=genre
        )

        if features:
            fallback_sentiment = {
                "polarity": (features.get("valence", 0.5) - 0.5) * 2,
                "subjectivity": features.get("energy", 0.5),
                "language": "unknown",
                "translated": False,
                "lyrics_found": False,
                "gemini_fallback": True,
                "features_used": True,
                "tags": features.get("tags", [])
            }
            print(f"✅ Gemini fallback succeeded for {track_name}")
            await cache_service.cache_lyrics_sentiment(track_name, artist_name, fallback_sentiment)
            return fallback_sentiment

        else:
            print(f"⚠️ Gemini AI could not generate fallback sentiment")
            neutral = {
                "polarity": 0.0,
                "subjectivity": 0.0,
                "language": "unknown",
                "translated": False,
                "lyrics_found": False,
                "gemini_fallback": False
            }
            await cache_service.cache_lyrics_sentiment(track_name, artist_name, neutral)
            return neutral

    except Exception as e:
        print(f"❌ Gemini fallback failed for {track_name}: {e}")
        neutral = {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "language": "unknown",
            "translated": False,
            "lyrics_found": False,
            "gemini_fallback": False,
            "error": error or str(e)
        }
        await cache_service.cache_lyrics_sentiment(track_name, artist_name, neutral)
        return neutral


# ============================================================
# Batch lyrics sentiment fetch
# ============================================================

async def batch_get_lyrics_sentiment(
    tracks: list,
    max_concurrent: int = 5
) -> Dict[str, Dict]:
    """
    Fetch lyrics sentiment for multiple tracks concurrently.
    """
    import asyncio
    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_semaphore(track):
        async with semaphore:
            key = f"{track['name']}_{track['artist']}"
            sentiment = await get_lyrics_sentiment(track['name'], track['artist'])
            return key, sentiment

    print(f"🎵 Fetching lyrics for {len(tracks)} tracks (max {max_concurrent} concurrent)...")
    completed = await asyncio.gather(*[fetch_with_semaphore(t) for t in tracks], return_exceptions=True)

    for r in completed:
        if isinstance(r, Exception):
            print(f"❌ Error in batch fetch: {r}")
            continue
        key, sentiment = r
        results[key] = sentiment

    print(f"✅ Completed batch lyrics fetch: {len(results)}/{len(tracks)} successful")
    return results


# ============================================================
# Mood interpretation + weighting helpers
# ============================================================

def get_mood_from_lyrics(sentiment: Dict) -> str:
    """
    Determine mood category from lyrics sentiment.
    """
    polarity = sentiment.get('polarity', 0.0)
    subjectivity = sentiment.get('subjectivity', 0.0)
    if subjectivity > 0.5:
        if polarity > 0.2:
            return "Positive"
        elif polarity < -0.2:
            return "Negative"
        else:
            return "Mixed"
    else:
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"


def calculate_lyrics_weight(sentiment: Dict) -> float:
    """
    Calculate how much weight lyrics should have in mood prediction.
    """
    polarity = abs(sentiment.get('polarity', 0.0))
    subjectivity = sentiment.get('subjectivity', 0.0)
    strength = (polarity * 0.7) + (subjectivity * 0.3)
    weight = 0.2 + (strength * 0.6)
    return min(weight, 0.8)


def get_sentiment_strength(sentiment: Dict) -> float:
    """
    Calculate overall sentiment strength for adaptive fusion weighting.
    """
    polarity = abs(sentiment.get('polarity', 0.0))
    subjectivity = sentiment.get('subjectivity', 0.0)
    return min(polarity * subjectivity, 1.0)


# ============================================================
# Testing
# ============================================================

async def test_lyrics_service():
    """
    Test the lyrics service with sample tracks.
    """
    print("\n" + "=" * 60)
    print("🧪 Testing Lyrics Service (with Gemini Fallback)")
    print("=" * 60)

    test_tracks = [
        {"name": "Happy", "artist": "Pharrell Williams"},
        {"name": "Someone Like You", "artist": "Adele"},
        {"name": "Nonexistent Song", "artist": "Unknown Artist"},  # To test fallback
    ]

    for track in test_tracks:
        print(f"\n🔍 Testing: {track['name']} by {track['artist']}")
        sentiment = await get_lyrics_sentiment(track['name'], track['artist'])
        print(f"   Polarity: {sentiment['polarity']:.3f}")
        print(f"   Subjectivity: {sentiment['subjectivity']:.3f}")
        print(f"   Language: {sentiment.get('language', 'unknown')}")
        print(f"   Translated: {sentiment.get('translated', False)}")
        print(f"   Lyrics found: {sentiment.get('lyrics_found', False)}")
        if sentiment.get("gemini_fallback"):
            print("   ⚡ Gemini Fallback Used")
        print(f"   Mood: {get_mood_from_lyrics(sentiment)}")
        print(f"   Weight: {calculate_lyrics_weight(sentiment):.3f}")
        print(f"   Strength: {get_sentiment_strength(sentiment):.3f}")

    print("\n✅ Testing complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_lyrics_service())
