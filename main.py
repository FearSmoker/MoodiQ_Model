from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import uvicorn
from datetime import datetime

# Import routers - FIXED: Import the router objects directly
from endpoints.mood_router import router as mood_router
from endpoints.optimize_router import router as optimize_router
from endpoints.train_router import router as train_router
from services import cache_service, model_service

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Moodify-AI ML Service",
    description="Machine Learning service for mood prediction, playlist optimization, and personalization",
    version="1.0.0",
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


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    """
    print("🚀 Starting Moodify-AI ML Service...")
    print(f"📅 Startup time: {datetime.utcnow().isoformat()}")
    
    # Connect to Redis
    try:
        await cache_service.connect_redis()
        print("✅ Redis connected")
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("   Service will run without caching")
    
    # Load the ML model
    try:
        model_service.load_model()
        if model_service.mood_model:
            print("✅ ML model loaded successfully")
        else:
            print("⚠️ ML model not loaded - using rule-based fallback")
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("   Service will use rule-based mood prediction")
    
    print("✅ Moodify-AI ML Service ready!")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    print("👋 Shutting down Moodify-AI ML Service...")
    
    try:
        await cache_service.disconnect_redis()
        print("✅ Redis disconnected")
    except Exception as e:
        print(f"⚠️ Error during Redis disconnect: {e}")
    
    print("✅ Shutdown complete")


# Include routers - FIXED: Removed .router attribute access
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
    tags=["Model Training & Feedback"]
)


# Root endpoint
@app.get("/")
def read_root():
    """
    Root endpoint with service information
    """
    return {
        "service": "Moodify-AI ML Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "mood_prediction": "/predict",
            "optimization": "/optimize",
            "training": "/model"
        },
        "features": [
            "Mood Prediction (ONNX Model + Rule-based Fallback)",
            "Lyrics Sentiment Fusion",
            "Multi-language Support",
            "Dynamic Programming Flow Optimization",
            "Greedy & Simulated Annealing Algorithms",
            "User Feedback Learning",
            "Personalized Recommendations",
            "Gap Detection",
            "Real-time Caching"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Comprehensive health check
    """
    redis_status = cache_service.redis_client is not None
    model_status = model_service.mood_model is not None
    
    # Determine overall health
    if redis_status and model_status:
        status = "healthy"
        http_status = 200
    elif model_status:
        status = "degraded"  # Can work without Redis
        http_status = 200
    else:
        status = "critical"  # No model loaded
        http_status = 503
    
    health_data = {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "redis": {
                "status": "connected" if redis_status else "disconnected",
                "available": redis_status
            },
            "ml_model": {
                "status": "loaded" if model_status else "not_loaded",
                "available": model_status,
                "type": "ONNX" if model_status else "rule_based_fallback"
            }
        },
        "uptime": os.popen('uptime -p').read().strip() if os.name != 'nt' else "N/A"
    }
    
    return JSONResponse(content=health_data, status_code=http_status)


@app.get("/stats")
async def get_service_stats():
    """
    Get service statistics
    """
    try:
        # Get cache stats if Redis is available
        cache_stats = {}
        if cache_service.redis_client:
            try:
                cache_stats = await cache_service.get_cache_stats()
            except:
                cache_stats = {"error": "Could not fetch cache stats"}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": model_service.mood_model is not None,
            "cache": cache_stats,
            "supported_moods": model_service.MOOD_CLASSES,
            "supported_algorithms": [
                "dynamic_programming",
                "greedy", 
                "simulated_annealing"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NLP Command Processing Endpoint
@app.post("/nlp/command")
async def process_nlp_command(request: dict):
    """
    Process natural language commands for voice/chat interface.
    Matches backend's voice-command endpoint.
    """
    try:
        command = request.get('command', '').lower()
        context = request.get('context', {})
        user_id = request.get('user_id')
        
        if not command:
            raise HTTPException(status_code=400, detail="Command is required")
        
        print(f"🗣️ Processing NLP command: {command}")
        
        # Simple rule-based NLP (in production, use a proper NLP model)
        action = "unknown"
        parameters = {}
        response_text = "I didn't understand that command."
        
        # Analyze playlist
        if any(word in command for word in ['analyze', 'check', 'what is', 'mood of']):
            action = "analyze_playlist"
            response_text = "I'll analyze the mood of your playlist."
        
        # Optimize playlist
        elif any(word in command for word in ['optimize', 'improve', 'reorder', 'flow']):
            action = "optimize_playlist"
            parameters = {"algorithm": "dynamic_programming"}
            response_text = "I'll optimize your playlist for smooth mood transitions."
        
        # Create playlist
        elif any(word in command for word in ['create', 'make', 'generate']):
            action = "create_playlist"
            
            # Extract mood from command
            if 'happy' in command or 'upbeat' in command:
                parameters = {"target_mood": "Happy"}
                response_text = "I'll create a happy, upbeat playlist for you."
            elif 'sad' in command or 'melancholy' in command:
                parameters = {"target_mood": "Sad"}
                response_text = "I'll create a melancholic playlist."
            elif 'calm' in command or 'relaxing' in command or 'chill' in command:
                parameters = {"target_mood": "Calm"}
                response_text = "I'll create a calm, relaxing playlist."
            elif 'energetic' in command or 'workout' in command or 'gym' in command:
                parameters = {"target_mood": "Energetic"}
                response_text = "I'll create an energetic workout playlist."
            else:
                response_text = "I'll create a playlist for you. What mood are you in?"
        
        # Transfer playlist
        elif any(word in command for word in ['transfer', 'export', 'move', 'copy']):
            action = "transfer_playlist"
            
            if 'youtube' in command:
                parameters = {"platform": "youtube"}
                response_text = "I'll transfer your playlist to YouTube Music."
            elif 'apple' in command:
                parameters = {"platform": "apple"}
                response_text = "I'll transfer your playlist to Apple Music."
            else:
                response_text = "Which platform would you like to transfer to?"
        
        # Get recommendations
        elif any(word in command for word in ['recommend', 'suggest', 'similar']):
            action = "get_recommendations"
            response_text = "I'll find some recommendations for you."
        
        # Help command
        elif 'help' in command:
            action = "help"
            response_text = ("I can help you with: analyzing playlists, optimizing song order, "
                           "creating mood-based playlists, transferring to other platforms, "
                           "and getting recommendations.")
        
        return {
            "success": True,
            "action": action,
            "parameters": parameters,
            "response": response_text,
            "confidence": 0.85 if action != "unknown" else 0.2
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ NLP processing error: {e}")
        return {
            "success": False,
            "action": "error",
            "parameters": {},
            "response": "I'm having trouble understanding that. Could you rephrase?",
            "error": str(e)
        }


# Error handlers
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


# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )