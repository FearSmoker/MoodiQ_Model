from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import uvicorn
from datetime import datetime

# Import routers
from endpoints.mood_router import router as mood_router
from endpoints.optimize_router import router as optimize_router
from endpoints.train_router import router as train_router
from services import cache_service, model_service, nlp_service
from endpoints.generate_router import router as generate_router 
from endpoints.analytics_router import router as analytics_router

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


# Include routers
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

app.include_router(
    generate_router,
    prefix="/generate",
    tags=["Generation"]
)
app.include_router(
    analytics_router,
    prefix="/analytics",
    tags=["Analytics"]
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
            "training": "/model",
            "nlp": "/nlp/command"
        },
        "features": [
            "Mood Prediction (ONNX Model + Rule-based Fallback)",
            "Advanced NLP (HuggingFace Offloaded)",
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
            },
            "nlp": {
                "status": "available",
                "provider": "huggingface_api",
                "fallback": "rule_based"
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
            ],
            "nlp": {
                "provider": "huggingface_api",
                "model": "facebook/bart-large-mnli",
                "fallback": "rule_based"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ADVANCED NLP ENDPOINT (UPDATED)
# ============================================
@app.post("/nlp/command")
async def process_nlp_command(request: dict):
    """
    Process natural language commands using HuggingFace NLP.
    Automatically falls back to rule-based if API unavailable.
    """
    try:
        command = request.get('command', '').strip()
        context = request.get('context', {})
        user_id = request.get('user_id')
        
        if not command:
            raise HTTPException(status_code=400, detail="Command is required")
        
        print(f"🗣️ Processing NLP command: {command}")
        
        # Use advanced NLP service
        result = await nlp_service.process_command_advanced(command, context)
        
        # Add user_id to result
        result['user_id'] = user_id
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ NLP processing error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "action": "error",
            "parameters": {},
            "response": "I'm having trouble understanding that. Could you rephrase?",
            "confidence": 0.0,
            "error": str(e),
            "method": "error"
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