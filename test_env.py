import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

print("=== Environment Variables Test ===\n")

# Test all API keys
api_keys = {
    "LASTFM_API_KEY": os.getenv("LASTFM_API_KEY"),
    "API_KEY_LASTFM": os.getenv("API_KEY_LASTFM"),
    "API_SECRET_LASTFM": os.getenv("API_SECRET_LASTFM"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "GENIUS_API_KEY": os.getenv("GENIUS_API_KEY"),
    "SPOTIFY_CLIENT_ID": os.getenv("SPOTIFY_CLIENT_ID"),
    "SPOTIFY_CLIENT_SECRET": os.getenv("SPOTIFY_CLIENT_SECRET"),
}

for key, value in api_keys.items():
    if value:
        # Show first 8 chars for security
        masked = f"{value[:8]}..." if len(value) > 8 else value
        print(f"✅ {key}: {masked}")
    else:
        print(f"❌ {key}: NOT FOUND")

print("\n=== Testing Service Initialization ===\n")

# Test services
from services import music_service, gemini_service

# Force initialization
music_service._init_lastfm()
gemini_service._init_gemini()

print(f"\nLast.fm Status: {'✅ Configured' if music_service.LASTFM_API_KEY else '❌ Not configured'}")
print(f"Gemini Status: {'✅ Configured' if gemini_service.GEMINI_API_KEY else '❌ Not configured'}")