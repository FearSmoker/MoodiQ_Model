"""
Redis caching service for MoodiQ-AI ML Service.

Provides caching for:
- Audio features
- Lyrics sentiment
- Mood predictions
- User overrides
- User statistics
"""

import redis.asyncio as redis
import os
import json
from typing import Optional, Any, Dict


# Global Redis client
redis_client: Optional[redis.Redis] = None


async def connect_redis():
    """
    Connect to Redis server with error handling and retry logic.
    Supports both local Redis and Upstash Redis.
    Requires redis>=5.0.1 for asyncio support.
    """
    global redis_client
    
    # Check if caching is enabled
    if os.getenv('ENABLE_CACHE', 'true').lower() != 'true':
        print("ℹ️  Caching is disabled (ENABLE_CACHE=false)")
        redis_client = None
        return None
    
    try:
        # Get Redis configuration from environment
        redis_url = os.getenv('REDIS_URL')
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_password = os.getenv('REDIS_PASSWORD', None)
        
        # Prefer REDIS_URL if provided (for Upstash and other cloud Redis)
        if redis_url:
            print(f"🔄 Connecting to Redis using REDIS_URL...")
            # Parse host from URL for display purposes
            if 'upstash.io' in redis_url:
                print(f"   Provider: Upstash Redis (Cloud)")
            else:
                print(f"   Using connection URL")
        else:
            # Construct Redis URL from components
            print(f"🔄 Connecting to Redis at {redis_host}:{redis_port}...")
            if redis_password:
                redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
            else:
                redis_url = f"redis://{redis_host}:{redis_port}"
        
        # Create Redis client with connection pool
        # Upstash-compatible settings
        redis_client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=10,  # Increased for cloud Redis
            socket_timeout=10,  # Increased for cloud Redis
            socket_keepalive=True,
            health_check_interval=30,
            max_connections=20,  # Lower for cloud Redis to avoid connection limits
            retry_on_timeout=True,  # Auto-retry on timeout
            retry_on_error=[redis.ConnectionError, redis.TimeoutError]  # Auto-retry on errors
        )
        
        # Test connection
        await redis_client.ping()
        
        # Get Redis info
        try:
            info = await redis_client.info('server')
            redis_version = info.get('redis_version', 'unknown')
            print(f"✅ Connected to Redis successfully")
            print(f"   Redis Server Version: {redis_version}")
        except Exception:
            # Upstash might restrict INFO command
            print(f"✅ Connected to Redis successfully")
            print(f"   (Cloud Redis - INFO command restricted)")
        
        return redis_client
        
    except redis.ConnectionError as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Service will continue without caching")
        redis_client = None
        return None
        
    except redis.TimeoutError as e:
        print(f"❌ Redis connection timeout: {e}")
        print("   Check your network connection and Redis URL")
        redis_client = None
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error connecting to Redis: {e}")
        print(f"   Error type: {type(e).__name__}")
        redis_client = None
        return None


async def disconnect_redis():
    """
    Gracefully disconnect from Redis.
    """
    global redis_client
    
    if redis_client:
        try:
            await redis_client.close()
            print("✅ Disconnected from Redis")
        except Exception as e:
            print(f"⚠️ Error disconnecting from Redis: {e}")
        finally:
            redis_client = None


async def get_from_cache(key: str) -> Optional[Any]:
    """
    Retrieve a value from cache by key.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value (parsed from JSON) or None if not found/error
    """
    if not redis_client:
        return None
    
    try:
        data = await redis_client.get(key)
        
        if data:
            print(f"📦 Cache HIT: {key}")
            
            # Try to parse as JSON
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return data
        else:
            print(f"🔍 Cache MISS: {key}")
            return None
            
    except redis.RedisError as e:
        print(f"⚠️ Redis error getting key '{key}': {e}")
        return None
        
    except Exception as e:
        print(f"⚠️ Error getting from cache '{key}': {e}")
        return None


async def set_in_cache(
    key: str, 
    value: Any, 
    expiration: int = 3600
) -> bool:
    """
    Store a value in cache with optional expiration.
    
    Args:
        key: Cache key
        value: Value to cache (will be JSON-serialized)
        expiration: TTL in seconds (default: 1 hour)
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        # Serialize value to JSON
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value)
        elif isinstance(value, str):
            serialized = value
        else:
            serialized = json.dumps(value)
        
        # Set with expiration
        await redis_client.set(key, serialized, ex=expiration)
        
        print(f"💾 Cache SET: {key} (expires in {expiration}s)")
        return True
        
    except redis.RedisError as e:
        print(f"⚠️ Redis error setting key '{key}': {e}")
        return False
        
    except Exception as e:
        print(f"⚠️ Error setting in cache '{key}': {e}")
        return False


async def delete_from_cache(key: str) -> bool:
    """
    Delete a key from cache.
    
    Args:
        key: Cache key to delete
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        deleted = await redis_client.delete(key)
        
        if deleted:
            print(f"🗑️ Cache DELETE: {key}")
            return True
        else:
            print(f"⚠️ Key not found for deletion: {key}")
            return False
            
    except redis.RedisError as e:
        print(f"⚠️ Redis error deleting key '{key}': {e}")
        return False
        
    except Exception as e:
        print(f"⚠️ Error deleting from cache '{key}': {e}")
        return False


async def delete_pattern(pattern: str) -> int:
    """
    Delete all keys matching a pattern.
    
    Args:
        pattern: Redis pattern (e.g., "user_model:user_123:*")
        
    Returns:
        Number of keys deleted
    """
    if not redis_client:
        return 0
    
    try:
        # Scan for keys matching pattern
        keys = []
        cursor = 0
        
        while True:
            cursor, batch = await redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            keys.extend(batch)
            
            if cursor == 0:
                break
        
        if keys:
            deleted = await redis_client.delete(*keys)
            print(f"🗑️ Cache DELETE pattern: {pattern} ({deleted} keys)")
            return deleted
        else:
            print(f"🔍 No keys found matching pattern: {pattern}")
            return 0
            
    except redis.RedisError as e:
        print(f"⚠️ Redis error deleting pattern '{pattern}': {e}")
        return 0
        
    except Exception as e:
        print(f"⚠️ Error deleting pattern '{pattern}': {e}")
        return 0


async def exists(key: str) -> bool:
    """
    Check if a key exists in cache.
    
    Args:
        key: Cache key
        
    Returns:
        True if key exists, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        result = await redis_client.exists(key)
        return bool(result)
        
    except Exception as e:
        print(f"⚠️ Error checking key existence '{key}': {e}")
        return False


async def get_ttl(key: str) -> int:
    """
    Get the remaining TTL (time to live) for a key.
    
    Args:
        key: Cache key
        
    Returns:
        TTL in seconds, -1 if no expiration, -2 if key doesn't exist
    """
    if not redis_client:
        return -2
    
    try:
        ttl = await redis_client.ttl(key)
        return ttl
        
    except Exception as e:
        print(f"⚠️ Error getting TTL for '{key}': {e}")
        return -2


async def extend_ttl(key: str, additional_seconds: int) -> bool:
    """
    Extend the TTL of an existing key.
    
    Args:
        key: Cache key
        additional_seconds: Seconds to add to current TTL
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        current_ttl = await redis_client.ttl(key)
        
        if current_ttl > 0:
            new_ttl = current_ttl + additional_seconds
            await redis_client.expire(key, new_ttl)
            print(f"⏰ Extended TTL for {key} by {additional_seconds}s")
            return True
        else:
            print(f"⚠️ Cannot extend TTL for {key} (TTL: {current_ttl})")
            return False
            
    except Exception as e:
        print(f"⚠️ Error extending TTL for '{key}': {e}")
        return False


async def increment(key: str, amount: int = 1) -> Optional[int]:
    """
    Increment a numeric value in cache.
    
    Args:
        key: Cache key
        amount: Amount to increment by (default: 1)
        
    Returns:
        New value after increment, or None on error
    """
    if not redis_client:
        return None
    
    try:
        new_value = await redis_client.incrby(key, amount)
        return new_value
        
    except Exception as e:
        print(f"⚠️ Error incrementing '{key}': {e}")
        return None


async def get_cache_stats() -> Dict[str, Any]:
    """
    Get Redis cache statistics and information.
    Compatible with both local Redis and Upstash (which may restrict INFO commands).
    
    Returns:
        Dictionary with cache statistics
    """
    if not redis_client:
        return {
            "connected": False,
            "error": "Redis not connected"
        }
    
    try:
        # Get database size (works on all Redis providers)
        db_size = await redis_client.dbsize()
        
        stats = {
            "connected": True,
            "db_size": db_size,
        }
        
        # Try to get detailed stats (may fail on Upstash/cloud Redis)
        try:
            info_stats = await redis_client.info('stats')
            info_memory = await redis_client.info('memory')
            info_clients = await redis_client.info('clients')
            
            # Calculate hit rate if available
            hits = info_stats.get('keyspace_hits', 0)
            misses = info_stats.get('keyspace_misses', 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            stats.update({
                "memory_used": info_memory.get('used_memory_human', 'N/A'),
                "memory_peak": info_memory.get('used_memory_peak_human', 'N/A'),
                "connected_clients": info_clients.get('connected_clients', 0),
                "total_commands": info_stats.get('total_commands_processed', 0),
                "keyspace_hits": hits,
                "keyspace_misses": misses,
                "hit_rate_percent": round(hit_rate, 2),
                "evicted_keys": info_stats.get('evicted_keys', 0),
                "expired_keys": info_stats.get('expired_keys', 0),
                "detailed_stats": True
            })
        except Exception:
            # Cloud Redis (Upstash) may restrict INFO command
            stats["detailed_stats"] = False
            stats["note"] = "Detailed stats unavailable (cloud Redis restrictions)"
        
        return stats
        
    except Exception as e:
        print(f"⚠️ Error getting cache stats: {e}")
        return {
            "connected": True,
            "error": str(e)
        }


async def flush_cache() -> bool:
    """
    Flush all keys from the current database.
    ⚠️ Use with caution - this deletes ALL cached data!
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        await redis_client.flushdb()
        print("🧹 Cache flushed - all keys deleted")
        return True
        
    except Exception as e:
        print(f"⚠️ Error flushing cache: {e}")
        return False


async def get_keys_by_pattern(pattern: str, limit: int = 100) -> list:
    """
    Get keys matching a pattern (for debugging/monitoring).
    
    Args:
        pattern: Redis pattern (e.g., "user_model:*")
        limit: Maximum number of keys to return
        
    Returns:
        List of matching keys
    """
    if not redis_client:
        return []
    
    try:
        keys = []
        cursor = 0
        
        while len(keys) < limit:
            cursor, batch = await redis_client.scan(
                cursor=cursor,
                match=pattern,
                count=100
            )
            keys.extend(batch)
            
            if cursor == 0:
                break
        
        return keys[:limit]
        
    except Exception as e:
        print(f"⚠️ Error getting keys by pattern '{pattern}': {e}")
        return []


async def set_json(key: str, value: Dict, expiration: int = 3600) -> bool:
    """
    Store a dictionary as JSON in cache (explicit JSON storage).
    
    Args:
        key: Cache key
        value: Dictionary to store
        expiration: TTL in seconds
        
    Returns:
        True if successful, False otherwise
    """
    if not isinstance(value, dict):
        print(f"⚠️ set_json requires a dictionary, got {type(value)}")
        return False
    
    return await set_in_cache(key, value, expiration)


async def get_json(key: str) -> Optional[Dict]:
    """
    Retrieve a JSON object from cache (explicit JSON retrieval).
    
    Args:
        key: Cache key
        
    Returns:
        Dictionary or None if not found
    """
    result = await get_from_cache(key)
    
    if result and isinstance(result, dict):
        return result
    
    return None


# Convenience functions for common cache patterns

async def cache_audio_features(track_id: str, features: Dict) -> bool:
    """Cache audio features for a track."""
    key = f"features:{track_id}"
    ttl = int(os.getenv('CACHE_TTL_AUDIO_FEATURES', '86400'))  # 1 day
    return await set_in_cache(key, features, ttl)


async def get_cached_audio_features(track_id: str) -> Optional[Dict]:
    """Get cached audio features for a track."""
    key = f"features:{track_id}"
    return await get_from_cache(key)


async def cache_lyrics_sentiment(track_name: str, artist_name: str, sentiment: Dict) -> bool:
    """Cache lyrics sentiment analysis."""
    key = f"lyrics:{track_name}:{artist_name}"
    ttl = int(os.getenv('CACHE_TTL_LYRICS', '604800'))  # 1 week
    return await set_in_cache(key, sentiment, ttl)


async def get_cached_lyrics_sentiment(track_name: str, artist_name: str) -> Optional[Dict]:
    """Get cached lyrics sentiment."""
    key = f"lyrics:{track_name}:{artist_name}"
    return await get_from_cache(key)


async def cache_mood_prediction(track_id: str, user_id: str, mood_data: Dict) -> bool:
    """Cache mood prediction for a track and user."""
    key = f"track:mood:{track_id}:{user_id}"
    ttl = int(os.getenv('CACHE_TTL_MOOD', '3600'))  # 1 hour
    return await set_in_cache(key, mood_data, ttl)


async def get_cached_mood_prediction(track_id: str, user_id: str) -> Optional[Dict]:
    """Get cached mood prediction."""
    key = f"track:mood:{track_id}:{user_id}"
    return await get_from_cache(key)


def is_connected() -> bool:
    """
    Check if Redis client is connected.
    
    Returns:
        True if connected, False otherwise
    """
    return redis_client is not None


# Export the client for direct access if needed
__all__ = [
    'connect_redis',
    'disconnect_redis',
    'get_from_cache',
    'set_in_cache',
    'delete_from_cache',
    'delete_pattern',
    'exists',
    'get_ttl',
    'extend_ttl',
    'increment',
    'get_cache_stats',
    'flush_cache',
    'get_keys_by_pattern',
    'set_json',
    'get_json',
    'cache_audio_features',
    'get_cached_audio_features',
    'cache_lyrics_sentiment',
    'get_cached_lyrics_sentiment',
    'cache_mood_prediction',
    'get_cached_mood_prediction',
    'is_connected',
    'redis_client'
]