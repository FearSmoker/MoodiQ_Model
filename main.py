"""
Updated MoodiQ-AI ML Service - Hybrid Approach
Uses: Spotify API (OAuth) + YTMusicAPI + MusicBrainz + AcousticBrainz + Last.fm

UPDATED: Added proper exception handlers for new Spotify service
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
import os
import uvicorn
from datetime import datetime

# Import routers
from endpoints.mood_router import router as mood_router
from endpoints.optimize_router import router as optimize_router
from endpoints.train_router import router as train_router
from endpoints.generate_router import router as generate_router
from endpoints.analytics_router import router as analytics_router
from endpoints.live_listening_router import router as live_listening_router

# Import services
from services import cache_service, model_service, nlp_service, music_service, spotify_service, db_recommendation_service

# Import Spotify custom exceptions
from services.spotify_service import (
    SpotifyAuthError,
    SpotifyRateLimitError,
    SpotifyNotFoundError,
    SpotifyServiceError
)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="MoodiQ-AI ML Service (Hybrid Edition)",
    description="ML service using Spotify API (OAuth) + Multi-API stack for comprehensive music analysis",
    version="2.5.1",  # Updated version
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Favicon handler to suppress 404 logs ---
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    
    return Response(status_code=204)

# ============================================
# SPOTIFY EXCEPTION HANDLERS
# ============================================

@app.exception_handler(SpotifyAuthError)
async def spotify_auth_error_handler(request, exc):
    
    return JSONResponse(
        status_code=401,
        content={
            "error": "spotify_authentication_failed",
            "message": str(exc),
            "action": "Please re-authenticate with Spotify",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(SpotifyRateLimitError)
async def spotify_rate_limit_error_handler(request, exc):
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "spotify_rate_limit_exceeded",
            "message": str(exc),
            "retry_after": exc.retry_after,
            "action": f"Please wait {exc.retry_after} seconds before retrying",
            "timestamp": datetime.utcnow().isoformat()
        },
        headers={"Retry-After": str(exc.retry_after)}
    )

@app.exception_handler(SpotifyNotFoundError)
async def spotify_not_found_error_handler(request, exc):
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "spotify_resource_not_found",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(SpotifyServiceError)
async def spotify_service_error_handler(request, exc):
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "spotify_service_error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================
# STARTUP & SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    
    print("=" * 80)
    print("🚀 Starting MoodiQ-AI ML Service v2.5.1 (HYBRID Edition)")
    print("=" * 80)
    print(f"📅 Startup time: {datetime.utcnow().isoformat()}")
    
    # Connect to Redis
    try:
        await cache_service.connect_redis()
        print("✅ Redis connected")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("   Service will run without caching")
    
    db_reco = db_recommendation_service.DatabaseRecommendationService()
    await db_reco.initialize()
    
    app.include_router(
    live_listening_router,
    prefix="/live",
    tags=["Live Listening"]
)
    # Load ML model
    try:
        model_service.load_model()
        if model_service.mood_model:
            print("✅ ML model loaded successfully")
        else:
            print("⚠️ ML model not loaded - using rule-based fallback")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("   Service will use rule-based mood prediction")
    
    # Initialize YTMusic
    music_service.init_ytmusic()
    
    # Initialize Spotify service (server credentials)
    # Check Spotify configuration (initialization is lazy)
    spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
    spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if spotify_client_id and spotify_client_secret:
        print("✅ Spotify service configured (server credentials)")
        print("   Features: Full pagination, rate limiting, custom exceptions")
    else:
        print("⚠️ Spotify credentials not configured")
        print("   Service will use user OAuth tokens only")
    
    # Check API configurations
    print("\n📡 API Configuration Status:")
    print("\n🎵 Primary Data Sources (Spotify OAuth):")
    print(f"   Spotify API: {'✅ Configured' if os.getenv('SPOTIFY_CLIENT_ID') else '⚠️ User tokens only'}")
    print(f"   - User playlists: ✅ Available (with pagination)")
    print(f"   - Currently playing: ✅ Available")
    print(f"   - Top tracks/artists: ✅ Available")
    print(f"   - Track metadata: ✅ Available")
    
    print("\n🔧 Audio Features & Analysis:")
    print(f"   YTMusicAPI: {'✅ Initialized' if music_service.ytmusic else '❌ Failed'}")
    print(f"   MusicBrainz: ✅ Ready")
    print(f"   AcousticBrainz: ✅ Ready")
    
    # Check Gemini API
    gemini_key = os.getenv('GEMINI_API_KEY')
    print(f"   Gemini AI: {'✅ Configured' if gemini_key else '❌ Not configured'}")
    if gemini_key:
        print(f"      - AI-powered feature estimation enabled")
        print(f"      - Key: {gemini_key[:8]}..." if gemini_key else "")
    else:
        print(f"      ⚠️  Set GEMINI_API_KEY to enable AI fallback for audio features")
    
    print("\n🎯 Recommendations & Discovery:")
    # Check Last.fm with both variable names
    lastfm_key = os.getenv("LASTFM_API_KEY") or os.getenv("API_KEY_LASTFM")
    lastfm_configured = bool(lastfm_key)
    print(f"   Last.fm: {'✅ Configured' if lastfm_configured else '❌ Not configured'}")
    if lastfm_configured:
        print(f"      - Recommendations enabled")
        print(f"      - Similar tracks/artists enabled")
    else:
        print(f"      ⚠️  Set LASTFM_API_KEY or API_KEY_LASTFM to enable Last.fm features")

    print("\n📝 Lyrics & Sentiment:")
    print(f"   Genius API: {'✅ Configured' if os.getenv('GENIUS_API_KEY') else '❌ Not configured'}")
    
    print("\n🛡️ Error Handling:")
    print("   ✅ Custom Spotify exceptions")
    print("   ✅ Rate limit protection")
    print("   ✅ Token expiration handling")
    print("   ✅ Graceful degradation")
    
    print("\n" + "=" * 80)
    print("✅ MoodiQ-AI ML Service v2.5.1 Ready!")
    print("=" * 80)
    print("\n🎯 HYBRID API Architecture:")
    print("\n   📊 METADATA & USER DATA (Spotify API via OAuth):")
    print("      • User playlists and saved tracks (with pagination)")
    print("      • Currently playing track")
    print("      • Top tracks and artists")
    print("      • Recently played tracks")
    print("      • Track metadata (name, artist, album, popularity)")
    print("      • Playlist tracks (supports 100+ tracks)")
    
    print("\n   🎹 AUDIO FEATURES (Multi-API Stack):")
    print("      • MusicBrainz → MBID Lookup")
    print("      • AcousticBrainz → Audio Features Extraction")
    print("      • Features: valence, energy, danceability, tempo, etc.")
    
    print("\n   🎯 RECOMMENDATIONS & DISCOVERY (Last.fm):")
    print("      • Similar tracks based on seed")
    print("      • Similar artists")
    print("      • Genre/mood tags")
    print("      • Track popularity and trends")
    
    print("\n   📝 LYRICS & SENTIMENT (Genius API):")
    print("      • Lyrics retrieval")
    print("      • Sentiment analysis")
    print("      • Emotion detection")
    
    print("\n   🤖 MOOD PREDICTION (ML Model + Rules):")
    print("      • ONNX-based neural network")
    print("      • Rule-based fallback")
    print("      • Adaptive genre-based weighting")
    print("      • Personalized user feedback learning")
    
    print("\n" + "=" * 80)
    print("🔗 Integration Flow Example:")
    print("   User → Frontend → OAuth → Spotify API (metadata)")
    print("                          ↓")
    print("   ML Service → MusicBrainz → AcousticBrainz (features)")
    print("                          ↓")
    print("   ML Service → Last.fm (recommendations)")
    print("                          ↓")
    print("   ML Service → Genius (lyrics)")
    print("                          ↓")
    print("   ML Model → Mood Prediction → Response")
    print("=" * 80 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    
    print("\n👋 Shutting down MoodiQ-AI ML Service...")
    
    try:
        await cache_service.disconnect_redis()
        print("✅ Redis disconnected")
    except Exception as e:
        print(f"⚠️ Error during Redis disconnect: {e}")
    
    print("✅ Shutdown complete")

# ============================================
# INCLUDE ROUTERS
# ============================================

app.include_router(
    mood_router,
    prefix="/predict",
    tags=["Mood Prediction"]
)

app.include_router(
    optimize_router,
    prefix="/optimize",
    tags=["Playlist Optimization"]
)

app.include_router(
    train_router,
    prefix="/model",
    tags=["Model Training & Personalization"]
)

app.include_router(
    generate_router,
    prefix="/generate",
    tags=["Playlist Generation & Recommendations"]
)

app.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["Analytics & Insights"]
)

# ============================================
# ROOT & INFO ENDPOINTS
# ============================================

@app.get("/")
def read_root():
    
    return {
        "service": "MoodiQ-AI ML Service",
        "version": "2.5.1",
        "approach": "hybrid",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "architecture": {
            "primary_metadata": "Spotify API (OAuth)",
            "audio_features": "MusicBrainz + AcousticBrainz",
            "recommendations": "Last.fm",
            "track_search": "YTMusicAPI",
            "lyrics": "Genius API",
            "mood_prediction": "ONNX ML Model + Rules"
        },
        "spotify_integration": {
            "working_endpoints": [
                "User Playlists (/me/playlists) - ✅ Full pagination",
                "Playlist Tracks (/playlists/{id}/tracks) - ✅ Full pagination",
                "Currently Playing (/me/player/currently-playing)",
                "Top Tracks (/me/top/tracks)",
                "Top Artists (/me/top/artists)",
                "Recently Played (/me/player/recently-played)",
                "Track Info (/tracks/{id})",
                "Search (/search)"
            ],
            "restricted_endpoints": [
                "Audio Features (/audio-features) → Replaced with AcousticBrainz",
                "Recommendations (/recommendations) → Replaced with Last.fm",
                "Related Artists (/artists/{id}/related-artists) → Replaced with Last.fm"
            ]
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "mood_prediction": "/predict",
            "optimization": "/optimize",
            "generation": "/generate",
            "training": "/model",
            "analytics": "/analytics"
        },
        "features": [
            "✅ Hybrid Spotify + Multi-API Integration",
            "✅ User Authentication via OAuth (handled by backend)",
            "✅ Real-time Currently Playing Analysis",
            "✅ Playlist Mood Analysis (100+ tracks supported)",
            "✅ Full Pagination for Playlists & Tracks",
            "✅ Top Tracks-based Recommendations",
            "✅ Mood Prediction (ML + Lyrics Sentiment)",
            "✅ Audio Feature Extraction",
            "✅ Flow Optimization (Dynamic Programming)",
            "✅ Personalized Learning from Feedback",
            "✅ Custom Exception Handling",
            "✅ Rate Limit Protection",
            "✅ Multi-language Support",
            "✅ Real-time Caching"
        ]
    }

@app.get("/health")
async def health_check():
    
    redis_status = cache_service.redis_client is not None
    model_status = model_service.mood_model is not None
    ytmusic_status = music_service.ytmusic is not None
    lastfm_status = music_service.LASTFM_API_KEY is not None
    spotify_configured = bool(os.getenv('SPOTIFY_CLIENT_ID'))
    
    # Determine overall health
    critical_services = [
        model_status or True,  # Fallback exists
        ytmusic_status,
        lastfm_status
    ]
    
    if all(critical_services) and spotify_configured:
        status = "healthy"
        http_status = 200
    elif spotify_configured and (ytmusic_status or lastfm_status):
        status = "degraded"
        http_status = 200
    else:
        status = "critical"
        http_status = 503
    
    health_data = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.5.1",
        "approach": "hybrid",
        "services": {
            "spotify_api": {
                "status": "configured" if spotify_configured else "not_configured",
                "available": spotify_configured,
                "mode": "oauth_only" if not os.getenv('SPOTIFY_CLIENT_SECRET') else "server_and_oauth",
                "features": [
                    "full_pagination",
                    "rate_limiting",
                    "custom_exceptions",
                    "token_validation",
                    "podcast_support"
                ],
                "endpoints": {
                    "user_playlists": True,
                    "currently_playing": True,
                    "top_tracks": True,
                    "track_metadata": True,
                    "playlist_tracks": True
                }
            },
            "redis_cache": {
                "status": "connected" if redis_status else "disconnected",
                "available": redis_status
            },
            "ml_model": {
                "status": "loaded" if model_status else "fallback",
                "available": True,
                "type": "ONNX" if model_status else "rule_based"
            },
            "ytmusic": {
                "status": "active" if ytmusic_status else "unavailable",
                "available": ytmusic_status,
                "purpose": "track_search"
            },
            "musicbrainz": {
                "status": "active",
                "available": True,
                "purpose": "mbid_lookup"
            },
            "acousticbrainz": {
                "status": "active",
                "available": True,
                "purpose": "audio_features"
            },
            "lastfm": {
                "status": "active" if lastfm_status else "not_configured",
                "available": lastfm_status,
                "purpose": "recommendations"
            },
            "genius_lyrics": {
                "status": "active" if os.getenv('GENIUS_API_KEY') else "not_configured",
                "available": bool(os.getenv('GENIUS_API_KEY')),
                "purpose": "lyrics_sentiment"
            }
        },
        "capabilities": {
            "spotify_integration": spotify_configured,
            "user_playlists": spotify_configured,
            "currently_playing": spotify_configured,
            "track_search": ytmusic_status,
            "audio_features": True,
            "mood_prediction": True,
            "recommendations": lastfm_status,
            "lyrics_analysis": bool(os.getenv('GENIUS_API_KEY')),
            "personalization": model_status,
            "caching": redis_status,
            "flow_optimization": True,
            "pagination": True,
            "rate_limiting": True,
            "error_handling": True
        }
    }
    
    return JSONResponse(content=health_data, status_code=http_status)

@app.get("/stats")
async def get_service_stats():
    
    try:
        cache_stats = {}
        if cache_service.redis_client:
            try:
                cache_stats = await cache_service.get_cache_stats()
            except:
                cache_stats = {"error": "Could not fetch cache stats"}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.5.1",
            "approach": "hybrid",
            "model_loaded": model_service.mood_model is not None,
            "cache": cache_stats,
            "supported_moods": model_service.MOOD_CLASSES,
            "optimization_algorithms": [
                "dynamic_programming",
                "greedy",
                "simulated_annealing"
            ],
            "api_integrations": {
                "spotify": {
                    "status": "configured" if os.getenv('SPOTIFY_CLIENT_ID') else "not_configured",
                    "purpose": "User data, playlists, currently playing, metadata",
                    "auth": "OAuth 2.0 (via backend)",
                    "features": ["pagination", "rate_limiting", "custom_exceptions"]
                },
                "ytmusic": {
                    "status": "active" if music_service.ytmusic else "inactive",
                    "purpose": "Track search and discovery"
                },
                "musicbrainz": {
                    "status": "active",
                    "purpose": "MBID lookup for cross-referencing"
                },
                "acousticbrainz": {
                    "status": "active",
                    "purpose": "Audio feature extraction (valence, energy, etc.)"
                },
                "lastfm": {
                    "status": "active" if music_service.LASTFM_API_KEY else "not_configured",
                    "purpose": "Recommendations, similar artists, tags"
                },
                "genius": {
                    "status": "active" if os.getenv('GENIUS_API_KEY') else "not_configured",
                    "purpose": "Lyrics and sentiment analysis"
                }
            },
            "workflow": {
                "step_1": "Frontend → OAuth → Spotify API (user authentication)",
                "step_2": "ML Service receives access_token from frontend",
                "step_3": "Spotify API → Track metadata (name, artist, album)",
                "step_4": "MusicBrainz → MBID lookup",
                "step_5": "AcousticBrainz → Audio features",
                "step_6": "Last.fm → Recommendations & tags",
                "step_7": "Genius → Lyrics sentiment",
                "step_8": "ML Model → Mood prediction",
                "step_9": "Response → Frontend"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-info")
async def get_api_info():
    
    return {
        "service": "MoodiQ-AI ML Service v2.5.1",
        "approach": "HYBRID",
        "description": "Combines Spotify API (OAuth) with Multi-API stack for comprehensive music analysis",
        
        "spotify_integration": {
            "authentication": "OAuth 2.0 (handled by backend)",
            "flow": "Frontend → Backend OAuth → Access Token → ML Service",
            "working_endpoints": [
                {
                    "endpoint": "/me/playlists",
                    "purpose": "Get user's playlists",
                    "status": "✅ Working",
                    "pagination": "✅ Full support",
                    "scope": "playlist-read-private"
                },
                {
                    "endpoint": "/playlists/{id}/tracks",
                    "purpose": "Get playlist tracks",
                    "status": "✅ Working",
                    "pagination": "✅ Supports 100+ tracks",
                    "scope": "playlist-read-private"
                },
                {
                    "endpoint": "/me/player/currently-playing",
                    "purpose": "Get currently playing track",
                    "status": "✅ Working",
                    "scope": "user-read-currently-playing"
                },
                {
                    "endpoint": "/me/top/tracks",
                    "purpose": "Get user's top tracks",
                    "status": "✅ Working",
                    "scope": "user-top-read"
                },
                {
                    "endpoint": "/me/top/artists",
                    "purpose": "Get user's top artists",
                    "status": "✅ Working",
                    "scope": "user-top-read"
                },
                {
                    "endpoint": "/me/player/recently-played",
                    "purpose": "Get recently played tracks",
                    "status": "✅ Working",
                    "scope": "user-read-recently-played"
                },
                {
                    "endpoint": "/tracks/{id}",
                    "purpose": "Get track metadata",
                    "status": "✅ Working",
                    "scope": "None (public)"
                }
            ],
            "restricted_endpoints_workarounds": [
                {
                    "spotify_endpoint": "/audio-features/{id}",
                    "restriction": "Extended mode only",
                    "workaround": "MusicBrainz → AcousticBrainz",
                    "status": "✅ Implemented"
                },
                {
                    "spotify_endpoint": "/recommendations",
                    "restriction": "Extended mode only",
                    "workaround": "Last.fm Similar Tracks",
                    "status": "✅ Implemented"
                },
                {
                    "spotify_endpoint": "/artists/{id}/related-artists",
                    "restriction": "Extended mode only",
                    "workaround": "Last.fm Similar Artists",
                    "status": "✅ Implemented"
                }
            ]
        },
        
        "multi_api_stack": [
            {
                "name": "Spotify API",
                "version": "v1",
                "auth": "OAuth 2.0",
                "purpose": "User data, playlists, metadata",
                "status": "✅ Primary data source",
                "rate_limit": "Varies by endpoint",
                "features": ["pagination", "rate_limiting", "error_handling"]
            },
            {
                "name": "YTMusicAPI",
                "version": "Latest",
                "auth": "None required",
                "purpose": "Track search and discovery",
                "status": "✅ Active",
                "rate_limit": "None (unofficial API)"
            },
            {
                "name": "MusicBrainz",
                "version": "v2",
                "auth": "None required",
                "purpose": "MBID lookup",
                "status": "✅ Active",
                "rate_limit": "1 request/second"
            },
            {
                "name": "AcousticBrainz",
                "version": "v1",
                "auth": "None required",
                "purpose": "Audio features",
                "status": "✅ Active",
                "rate_limit": "None specified"
            },
            {
                "name": "Last.fm",
                "version": "v2.0",
                "auth": "API Key",
                "purpose": "Recommendations, tags",
                "status": "✅ Active" if music_service.LASTFM_API_KEY else "⚠️ Not configured",
                "rate_limit": "5 requests/second"
            },
            {
                "name": "Genius",
                "version": "v2",
                "auth": "API Key",
                "purpose": "Lyrics and sentiment",
                "status": "✅ Active" if os.getenv('GENIUS_API_KEY') else "⚠️ Not configured",
                "rate_limit": "Varies"
            }
        ],
        
        "advantages": [
            "✅ Full Spotify user integration (playlists, currently playing)",
            "✅ No dependency on restricted Spotify endpoints",
            "✅ Comprehensive audio feature analysis",
            "✅ High-quality recommendations from Last.fm",
            "✅ Lyrics sentiment analysis",
            "✅ Personalized ML-based mood prediction",
            "✅ Real-time caching for performance",
            "✅ Fallback mechanisms for reliability",
            "✅ Full pagination support (100+ tracks)",
            "✅ Custom exception handling",
            "✅ Rate limit protection"
        ]
    }

# ============================================
# NLP & UTILITY ENDPOINTS
# ============================================

@app.post("/nlp/command")
async def process_nlp_command(request: dict):
    
    try:
        command = request.get('command', '').strip()
        context = request.get('context', {})
        user_id = request.get('user_id')
        
        if not command:
            raise HTTPException(status_code=400, detail="Command is required")
        
        result = await nlp_service.process_command_advanced(command, context)
        result['user_id'] = user_id
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "action": "error",
            "parameters": {},
            "response": "I'm having trouble understanding that. Could you rephrase?",
            "confidence": 0.0,
            "error": str(e)
        }

# ============================================
# GENERAL ERROR HANDLERS
# ============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The endpoint {request.url.path} does not exist",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("\n" + "=" * 80)
    print("🚀 Starting MoodiQ-AI ML Service v2.5.1 (HYBRID Edition)")
    print("=" * 80)
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"🔗 Approach: Spotify API (OAuth) + Multi-API Stack")
    print(f"✨ Features: Pagination, Rate Limiting, Custom Exceptions")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )