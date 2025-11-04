"""
Advanced NLP service using HuggingFace Inference API (offloaded).
Fixed to use async HTTP calls to prevent blocking.
"""

import os
import httpx  # Changed from requests to httpx for async support
from typing import Dict, Tuple, Optional, List
from . import cache_service

# HuggingFace API Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_API_URL = os.getenv("HF_API_URL")

# Intent labels for zero-shot classification
INTENT_LABELS = [
    "create_playlist",
    "optimize_playlist", 
    "analyze_playlist",
    "transfer_playlist",
    "get_recommendations",
    "help"
]

# Mood labels for mood extraction
MOOD_LABELS = ["Happy", "Sad", "Calm", "Energetic"]

# Activity labels
ACTIVITY_LABELS = [
    "workout", "gym", "exercise",
    "study", "focus", "work",
    "party", "celebration",
    "sleep", "relax", "meditation",
    "driving", "commute"
]


async def classify_intent_hf(command: str) -> Tuple[str, float]:
    """
    Classify user intent using HuggingFace zero-shot classification.
    NOW PROPERLY ASYNC - won't block the event loop.
    
    Args:
        command: User's natural language command
        
    Returns:
        Tuple of (intent, confidence_score)
    """
    # Check cache first
    cache_key = f"nlp:intent:{command.lower()}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        print(f"📦 NLP Cache HIT: {command[:50]}")
        return cached['intent'], cached['confidence']
    
    print(f"🔍 Classifying intent via HuggingFace API...")
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        payload = {
            "inputs": command,
            "parameters": {
                "candidate_labels": INTENT_LABELS
            }
        }
        
        # Use httpx for async requests
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            if response.status_code == 503:
                # Model is loading on HF servers
                print("⏳ HuggingFace model is loading, retrying...")
                import asyncio
                await asyncio.sleep(2)
                response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            response.raise_for_status()
            result = response.json()
        
        # Extract top prediction
        intent = result['labels'][0]
        confidence = result['scores'][0]
        
        # Cache for 1 hour
        await cache_service.set_in_cache(
            cache_key,
            {"intent": intent, "confidence": confidence},
            expiration=3600
        )
        
        print(f"✅ Intent classified: {intent} (confidence: {confidence:.2f})")
        return intent, confidence
        
    except httpx.TimeoutException:
        print("⚠️ HuggingFace API timeout, falling back to rule-based")
        return fallback_intent_classification(command)
        
    except httpx.HTTPError as e:
        print(f"⚠️ HuggingFace API error: {e}, falling back to rule-based")
        return fallback_intent_classification(command)
        
    except Exception as e:
        print(f"⚠️ Unexpected NLP error: {e}")
        return fallback_intent_classification(command)


async def extract_mood_hf(command: str) -> Optional[str]:
    """
    Extract mood from command using zero-shot classification.
    NOW PROPERLY ASYNC.
    
    Args:
        command: User's command
        
    Returns:
        Detected mood or None
    """
    cache_key = f"nlp:mood:{command.lower()}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached.get('mood')
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        payload = {
            "inputs": command,
            "parameters": {
                "candidate_labels": MOOD_LABELS
            }
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            if response.status_code == 503:
                import asyncio
                await asyncio.sleep(2)
                response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            response.raise_for_status()
            result = response.json()
        
        mood = result['labels'][0]
        confidence = result['scores'][0]
        
        # Only return if confidence is high enough
        if confidence > 0.4:
            await cache_service.set_in_cache(
                cache_key,
                {"mood": mood, "confidence": confidence},
                expiration=3600
            )
            return mood
        
        return None
        
    except Exception as e:
        print(f"⚠️ Mood extraction error: {e}")
        return fallback_mood_extraction(command)


async def extract_activity_hf(command: str) -> Optional[str]:
    """
    Extract activity from command.
    NOW PROPERLY ASYNC.
    
    Args:
        command: User's command
        
    Returns:
        Detected activity or None
    """
    cache_key = f"nlp:activity:{command.lower()}"
    cached = await cache_service.get_from_cache(cache_key)
    
    if cached:
        return cached.get('activity')
    
    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        
        payload = {
            "inputs": command,
            "parameters": {
                "candidate_labels": ACTIVITY_LABELS
            }
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            if response.status_code == 503:
                import asyncio
                await asyncio.sleep(2)
                response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            response.raise_for_status()
            result = response.json()
        
        activity = result['labels'][0]
        confidence = result['scores'][0]
        
        if confidence > 0.5:
            await cache_service.set_in_cache(
                cache_key,
                {"activity": activity, "confidence": confidence},
                expiration=3600
            )
            return activity
        
        return None
        
    except Exception as e:
        print(f"⚠️ Activity extraction error: {e}")
        return fallback_activity_extraction(command)


def fallback_intent_classification(command: str) -> Tuple[str, float]:
    """
    Rule-based fallback when HuggingFace API is unavailable.
    """
    command_lower = command.lower()
    
    if any(word in command_lower for word in ['analyze', 'check', 'what is', 'mood of']):
        return "analyze_playlist", 0.7
    
    elif any(word in command_lower for word in ['optimize', 'improve', 'reorder', 'flow']):
        return "optimize_playlist", 0.7
    
    elif any(word in command_lower for word in ['create', 'make', 'generate']):
        return "create_playlist", 0.7
    
    elif any(word in command_lower for word in ['transfer', 'export', 'move', 'copy']):
        return "transfer_playlist", 0.7
    
    elif any(word in command_lower for word in ['recommend', 'suggest', 'similar']):
        return "get_recommendations", 0.7
    
    elif 'help' in command_lower:
        return "help", 0.9
    
    return "unknown", 0.2


def fallback_mood_extraction(command: str) -> Optional[str]:
    """
    Rule-based mood extraction fallback.
    """
    command_lower = command.lower()
    
    if any(word in command_lower for word in ['happy', 'upbeat', 'joyful', 'cheerful']):
        return "Happy"
    elif any(word in command_lower for word in ['sad', 'melancholy', 'depressing']):
        return "Sad"
    elif any(word in command_lower for word in ['calm', 'chill', 'relaxing', 'peaceful']):
        return "Calm"
    elif any(word in command_lower for word in ['energetic', 'workout', 'gym', 'hype']):
        return "Energetic"
    
    return None


def fallback_activity_extraction(command: str) -> Optional[str]:
    """
    Rule-based activity extraction fallback.
    """
    command_lower = command.lower()
    
    if any(word in command_lower for word in ['workout', 'gym', 'exercise']):
        return "workout"
    elif any(word in command_lower for word in ['study', 'focus', 'work']):
        return "study"
    elif any(word in command_lower for word in ['party', 'celebration']):
        return "party"
    elif any(word in command_lower for word in ['sleep', 'relax', 'meditation', 'yoga']):
        return "meditation"
    elif any(word in command_lower for word in ['driving', 'commute']):
        return "driving"
    
    return None


async def process_command_advanced(command: str, context: Dict = None) -> Dict:
    """
    Process natural language command with advanced NLP.
    NOW PROPERLY ASYNC - all HTTP calls are non-blocking.
    
    Args:
        command: User's natural language command
        context: Optional context (user_id, current_playlist, etc.)
        
    Returns:
        Structured command with intent, parameters, and response
    """
    print(f"🗣️ Processing advanced NLP: {command}")
    
    # 1. Classify intent
    intent, confidence = await classify_intent_hf(command)
    
    # 2. Extract entities
    mood = await extract_mood_hf(command)
    activity = await extract_activity_hf(command)
    
    # 3. Build parameters
    parameters = {}
    response_text = ""
    
    if intent == "create_playlist":
        if mood:
            parameters["target_mood"] = mood
            response_text = f"I'll create a {mood.lower()} playlist for you."
        elif activity:
            parameters["activity"] = activity
            response_text = f"I'll create a playlist for {activity}."
        else:
            response_text = "I'll create a playlist. What mood or activity?"
    
    elif intent == "optimize_playlist":
        parameters["algorithm"] = "dynamic_programming"
        response_text = "I'll optimize your playlist for smooth mood transitions."
    
    elif intent == "analyze_playlist":
        response_text = "I'll analyze the mood of your playlist."
    
    elif intent == "transfer_playlist":
        # Extract platform from command
        if 'youtube' in command.lower():
            parameters["platform"] = "youtube"
            response_text = "I'll transfer your playlist to YouTube Music."
        elif 'apple' in command.lower():
            parameters["platform"] = "apple"
            response_text = "I'll transfer your playlist to Apple Music."
        else:
            response_text = "Which platform would you like to transfer to?"
    
    elif intent == "get_recommendations":
        if mood:
            parameters["target_mood"] = mood
        response_text = "I'll find some recommendations for you."
    
    elif intent == "help":
        response_text = (
            "I can help you with: analyzing playlists, optimizing song order, "
            "creating mood-based playlists, transferring to other platforms, "
            "and getting recommendations."
        )
    
    else:
        response_text = "I didn't quite understand that. Could you rephrase?"
    
    return {
        "success": True,
        "action": intent,
        "parameters": parameters,
        "response": response_text,
        "confidence": float(confidence),
        "detected_mood": mood,
        "detected_activity": activity,
        "method": "huggingface_api" if confidence > 0.5 else "fallback"
    }


# Synonym mapping for better understanding
MOOD_SYNONYMS = {
    "joyful": "Happy",
    "cheerful": "Happy",
    "upbeat": "Happy",
    "excited": "Happy",
    "melancholic": "Sad",
    "depressing": "Sad",
    "gloomy": "Sad",
    "peaceful": "Calm",
    "chill": "Calm",
    "tranquil": "Calm",
    "hype": "Energetic",
    "pumped": "Energetic",
    "intense": "Energetic"
}

def expand_mood_with_synonyms(mood: Optional[str]) -> Optional[str]:
    """Map synonyms to standard mood classes."""
    if not mood:
        return None
    
    mood_lower = mood.lower()
    return MOOD_SYNONYMS.get(mood_lower, mood)