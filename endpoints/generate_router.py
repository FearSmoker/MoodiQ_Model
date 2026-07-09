"""
Updated Generate Router - 12 Moods Multi-Tag Compatible
All endpoints updated to work with extended mood system
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from services import music_service, model_service, lyrics_service, spotify_service, cache_service

router = APIRouter()

class GeneratePlaylistRequest(BaseModel):
    target_mood: str
    user_id: str
    seed_track_name: Optional[str] = None
    seed_artist_name: Optional[str] = None
    seed_track_id: Optional[str] = None
    limit: int = 20

class GenerateActivityRequest(BaseModel):
    activity: str
    user_id: str
    seed_track_name: Optional[str] = None
    seed_artist_name: Optional[str] = None
    seed_track_id: Optional[str] = None
    limit: int = 20

# ============================================

# ============================================

@router.post("/playlist")
async def generate_mood_playlist(
    request: GeneratePlaylistRequest,
    authorization: str = Header(None)
):
    
    try:
        # Map external mood to extended mood (12 moods)
        extended_mood = model_service.map_external_mood_to_extended(request.target_mood)
        
        print(f"🎯 Generating playlist")
        print(f"   Requested mood: {request.target_mood}")
        print(f"   Extended mood: {extended_mood}")
        
        # Determine seed track
        seed_track_name = request.seed_track_name
        seed_artist_name = request.seed_artist_name
        
        # If Spotify track ID provided, get track info from Spotify
        if request.seed_track_id and authorization and authorization.startswith('Bearer '):
            try:
                access_token = authorization.replace('Bearer ', '')
                track_info = await spotify_service.get_track_info(request.seed_track_id, access_token)
                
                if track_info:
                    seed_track_name = track_info['name']
                    seed_artist_name = spotify_service.get_primary_artist_name(track_info)
                    print(f"✅ Got seed from Spotify: {seed_track_name} by {seed_artist_name}")
            except Exception as e:
                print(f"⚠️ Could not get Spotify track info: {e}")
        
        # If still no seed, use defaults based on EXTENDED mood
        if not seed_track_name or not seed_artist_name:
            mood_seeds = {
                "Joyful": ("Happy", "Pharrell Williams"),
                "Excited": ("Can't Stop the Feeling", "Justin Timberlake"),
                "Party": ("Uptown Funk", "Mark Ronson"),
                "Melancholic": ("Someone Like You", "Adele"),
                "Dreamy": ("Breathe Me", "Sia"),
                "Relaxed": ("Weightless", "Marconi Union"),
                "Chill": ("Electric Feel", "MGMT"),
                "Focused": ("Clair de Lune", "Claude Debussy"),
                "Romantic": ("Thinking Out Loud", "Ed Sheeran"),
                "Motivated": ("Stronger", "Kanye West"),
                "Angry": ("Break Stuff", "Limp Bizkit"),
                "Ambient": ("Avril 14th", "Aphex Twin")
            }
            seed_track_name, seed_artist_name = mood_seeds.get(
                extended_mood, 
                ("Happy", "Pharrell Williams")
            )
        
        print(f"🌱 Using seed: {seed_track_name} by {seed_artist_name}")
        
        # Get base moods associated with this extended mood
        base_moods = model_service.EXTENDED_MOODS[extended_mood]["base_moods"]
        primary_base_mood = base_moods[0]  # Use first base mood for filtering
        
        recommendations = []
        spotify_used = False
        access_token = None  # Initialize so it's always defined
        
        if authorization and authorization.startswith('Bearer '):
            access_token = authorization.replace('Bearer ', '')
            
            # Find Spotify track ID for seed
            seed_id = request.seed_track_id
            if not seed_id and seed_track_name:
                try:
                    search_res = await spotify_service.search_tracks(
                        f"{seed_track_name} {seed_artist_name}",
                        limit=1,
                        access_token=access_token
                    )
                    if search_res:
                        seed_id = search_res[0]['id']
                except Exception as e:
                    print(f"⚠️ Search failed for seed {seed_track_name}: {e}")
            
            if seed_id:
                try:
                    print(f"🚀 Fetching Spotify recommendations using seed ID: {seed_id}")
                    profile = model_service.MOOD_PROFILES.get(extended_mood)
                    target_valence = sum(profile['valence']) / 2.0 if profile else 0.5
                    target_energy = sum(profile['energy']) / 2.0 if profile else 0.5
                    
                    spotify_recs = await spotify_service.get_recommendations(
                        seed_tracks=[seed_id],
                        target_valence=target_valence,
                        target_energy=target_energy,
                        limit=request.limit * 3,
                        access_token=access_token
                    )
                    
                    if spotify_recs:
                        # Batch get audio features
                        rec_ids = [r['id'] for r in spotify_recs]
                        features_list = await spotify_service.get_audio_features(rec_ids, access_token)
                        
                        for idx, track in enumerate(spotify_recs):
                            track_features = features_list[idx] if idx < len(features_list) else None
                            if track_features:
                                recommendations.append({
                                    'name': track['name'],
                                    'artist': track['artists'][0] if track['artists'] else 'Unknown Artist',
                                    'features': track_features,
                                    'id': track['id'],
                                    'external_url': track['external_url']
                                })
                        spotify_used = True
                        print(f"✅ Found {len(recommendations)} candidate tracks from Spotify Recommendations")
                except Exception as e:
                    print(f"⚠️ Spotify recommendations failed, falling back to Last.fm: {e}")
                    
        if not spotify_used:
            print("🔊 Using Last.fm candidate generation...")
            recommendations = await music_service.get_recommendations(
                seed_track_name=seed_track_name,
                seed_artist_name=seed_artist_name,
                target_mood=primary_base_mood,
                limit=request.limit * 2,
                access_token=access_token  # May be None — music_service handles this
            )

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations found. Try a different seed track."
            )
        
        # Filter and analyze with multi-mood support
        filtered_tracks = []
        
        for track in recommendations:
            try:
                # Already has features from get_recommendations
                features = track.get('features')
                
                if not features:
                    continue
                
                # Get lyrics sentiment
                lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                    track['name'],
                    track['artist']
                )
                
                # Predict mood with multi-tag support
                mood_data = await model_service.predict_mood_from_features(
                    features,
                    lyrics_sentiment,
                    user_id=request.user_id,
                    genre=None
                )
                
                # Check if any of the predicted moods match target extended mood
                all_moods = mood_data.get('all_moods', [mood_data.get('primary_mood')])
                
                if extended_mood in all_moods:
                    track['primary_mood'] = mood_data['primary_mood']
                    track['all_moods'] = all_moods
                    track['mood_scores'] = mood_data.get('mood_scores', {})
                    track['confidence'] = mood_data['confidence']
                    track['mood_details'] = mood_data
                    filtered_tracks.append(track)
                
                if len(filtered_tracks) >= request.limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error processing track: {e}")
                continue
        
        # If no exact matches, filter by similarity
        if not filtered_tracks:
            print("⚠️ No exact mood matches, using similarity threshold")
            for track in recommendations:
                try:
                    features = track.get('features')
                    if not features:
                        continue
                    
                    # Calculate similarity to target mood
                    similarity = model_service.calculate_mood_similarity(
                        features, 
                        extended_mood
                    )
                    
                    if similarity >= 0.60:  # 60% similarity threshold
                        lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                            track['name'],
                            track['artist']
                        )
                        
                        mood_data = await model_service.predict_mood_from_features(
                            features,
                            lyrics_sentiment,
                            user_id=request.user_id,
                            genre=None
                        )
                        
                        track['primary_mood'] = mood_data['primary_mood']
                        track['all_moods'] = mood_data.get('all_moods', [])
                        track['mood_scores'] = mood_data.get('mood_scores', {})
                        track['confidence'] = similarity
                        track['mood_details'] = mood_data
                        filtered_tracks.append(track)
                    
                    if len(filtered_tracks) >= request.limit:
                        break
                except Exception as e:
                    print(f"⚠️ Error in similarity check: {e}")
                    continue
        
        if not filtered_tracks:
            print("⚠️ No matches found, returning top candidates")
            filtered_tracks = recommendations[:request.limit]
        
        print(f"✅ Filtered to {len(filtered_tracks)} tracks matching {extended_mood}")
        
        # Calculate mood distribution
        mood_distribution = model_service.calculate_playlist_mood_distribution(filtered_tracks)
        
        # Optimize flow if we have enough tracks
        if len(filtered_tracks) > 2:
            # Use the extended mood profile for optimization
            target_profile = model_service.EXTENDED_MOODS[extended_mood]["profile"]
            
            optimization = model_service.optimize_flow_dp(
                filtered_tracks,
                target_profile,
                target_profile
            )
            
            ordered_tracks = [filtered_tracks[i] for i in optimization['optimizedOrder']]
            flow_score = optimization['flowScore']
        else:
            ordered_tracks = filtered_tracks
            flow_score = 1.0
        
        return {
            "requested_mood": request.target_mood,
            "target_mood": extended_mood,
            "mood_mapped": request.target_mood.lower() != extended_mood.lower(),
            "tracks": ordered_tracks,
            "total": len(ordered_tracks),
            "flow_score": flow_score,
            "mood_distribution": mood_distribution,
            "seed_track": {
                "name": seed_track_name,
                "artist": seed_artist_name,
                "id": request.seed_track_id
            },
            "source": "spotify_recommendations" if spotify_used else "lastfm_recommendations",

            "approach": "hybrid_multi_mood",
            "mood_system": "12_extended_moods"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating mood playlist: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/activity")
async def generate_activity_playlist(
    request: GenerateActivityRequest,
    authorization: str = Header(None)
):
    
    try:
        print(f"🏃 Generating playlist for activity: {request.activity}")
        
        # Updated activity profiles with extended moods
        activity_profiles = {
            "study": {
                "mood": "Focused",
                "seed": ("Clair de Lune", "Claude Debussy")
            },
            "workout": {
                "mood": "Motivated",
                "seed": ("Stronger", "Kanye West")
            },
            "gym": {
                "mood": "Motivated",
                "seed": ("Eye of the Tiger", "Survivor")
            },
            "party": {
                "mood": "Party",
                "seed": ("Uptown Funk", "Mark Ronson")
            },
            "sleep": {
                "mood": "Relaxed",
                "seed": ("Weightless", "Marconi Union")
            },
            "meditation": {
                "mood": "Ambient",
                "seed": ("Om Mani Padme Hum", "Imee Ooi")
            },
            "work": {
                "mood": "Focused",
                "seed": ("Music for Airports", "Brian Eno")
            },
            "focus": {
                "mood": "Focused",
                "seed": ("Porcelain", "Moby")
            },
            "driving": {
                "mood": "Excited",
                "seed": ("Life is a Highway", "Tom Cochrane")
            },
            "relax": {
                "mood": "Relaxed",
                "seed": ("Weightless", "Marconi Union")
            },
            "chill": {
                "mood": "Chill",
                "seed": ("Electric Feel", "MGMT")
            },
            "romantic": {
                "mood": "Romantic",
                "seed": ("Thinking Out Loud", "Ed Sheeran")
            },
            "angry": {
                "mood": "Angry",
                "seed": ("Break Stuff", "Limp Bizkit")
            },
            "sad": {
                "mood": "Melancholic",
                "seed": ("Someone Like You", "Adele")
            },
            "happy": {
                "mood": "Joyful",
                "seed": ("Happy", "Pharrell Williams")
            },
            "energetic": {
                "mood": "Excited",
                "seed": ("Can't Stop the Feeling", "Justin Timberlake")
            }
        }
        
        activity_lower = request.activity.lower()
        
        if activity_lower not in activity_profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown activity. Choose from: {', '.join(activity_profiles.keys())}"
            )
        
        profile = activity_profiles[activity_lower]
        
        if not request.seed_track_name:
            request.seed_track_name, request.seed_artist_name = profile['seed']
        
        mood_request = GeneratePlaylistRequest(
            target_mood=profile['mood'],
            user_id=request.user_id,
            seed_track_name=request.seed_track_name,
            seed_artist_name=request.seed_artist_name,
            seed_track_id=request.seed_track_id,
            limit=request.limit
        )
        
        result = await generate_mood_playlist(mood_request, authorization)
        
        result['activity'] = request.activity
        result['activity_profile'] = profile
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating activity playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# SPOTIFY-BASED GENERATION (MULTI-MOOD)
# ============================================

@router.post("/spotify/from-top-tracks")
async def generate_from_spotify_top_tracks(
    request: Dict[str, Any],
    authorization: str = Header(None)
):
    
    try:
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        user_id = request.get('user_id')
        target_mood_raw = request.get('target_mood')
        limit = request.get('limit', 20)
        time_range = request.get('time_range', 'medium_term')
        
        # Map target mood to extended mood if provided
        target_mood = None
        if target_mood_raw:
            target_mood = model_service.map_external_mood_to_extended(target_mood_raw)
            print(f"🎯 Target mood: {target_mood_raw} → {target_mood}")
        
        print(f"🎵 Generating playlist from user's top tracks")
        
        # Get user's top tracks from Spotify
        top_tracks = await spotify_service.get_user_top_tracks(
            access_token,
            time_range=time_range,
            limit=5
        )
        
        if not top_tracks:
            raise HTTPException(status_code=404, detail="No top tracks found")
        
        print(f"✅ Got {len(top_tracks)} top tracks from Spotify")
        
        # Generate recommendations from each top track
        all_recommendations = []
        
        for top_track in top_tracks[:3]:
            track_name = top_track['name']
            artist_name = top_track['artists'][0]['name']
            
            print(f"🌱 Getting recommendations for: {track_name}")
            
            recommendations = await music_service.get_similar_tracks_lastfm(
                track_name,
                artist_name,
                limit=10
            )
            
            all_recommendations.extend(recommendations)
        
        # Remove duplicates
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            key = f"{rec['name']}:{rec['artist']}"
            if key not in seen:
                seen.add(key)
                unique_recommendations.append(rec)
        
        print(f"✅ Got {len(unique_recommendations)} unique recommendations")
        
        # Analyze with multi-mood support
        analyzed_tracks = []
        
        for track in unique_recommendations:
            try:
                # Get audio features
                features = await music_service.get_audio_features(
                    track['name'],
                    track['artist']
                )
                
                if not features:
                    continue
                
                # Get lyrics sentiment
                lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                    track['name'],
                    track['artist']
                )
                
                # Predict mood with multi-tag support
                mood_data = await model_service.predict_mood_from_features(
                    features,
                    lyrics_sentiment,
                    user_id=user_id
                )
                
                # Filter by target mood if specified
                if target_mood:
                    all_moods = mood_data.get('all_moods', [mood_data.get('primary_mood')])
                    if target_mood not in all_moods:
                        continue
                
                track['features'] = features
                track['primary_mood'] = mood_data['primary_mood']
                track['all_moods'] = mood_data.get('all_moods', [])
                track['mood_scores'] = mood_data.get('mood_scores', {})
                track['confidence'] = mood_data['confidence']
                track['mood_details'] = mood_data
                
                analyzed_tracks.append(track)
                
                if len(analyzed_tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error analyzing track: {e}")
                continue
        
        # Calculate mood distribution
        mood_distribution = model_service.calculate_playlist_mood_distribution(analyzed_tracks)
        
        return {
            "source": "spotify_top_tracks",
            "top_tracks_used": [
                {"name": t['name'], "artists": [a['name'] for a in t['artists']]}
                for t in top_tracks[:3]
            ],
            "tracks": analyzed_tracks,
            "total": len(analyzed_tracks),
            "requested_mood": target_mood_raw,
            "target_mood": target_mood,
            "mood_distribution": mood_distribution,
            "approach": "hybrid_multi_mood",
            "mood_system": "12_extended_moods"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating from top tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/spotify/from-recently-played")
async def generate_from_recently_played(
    request: Dict[str, Any],
    authorization: str = Header(None)
):
    
    try:
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        
        access_token = authorization.replace('Bearer ', '')
        user_id = request.get('user_id')
        target_mood_raw = request.get('target_mood')
        limit = request.get('limit', 20)
        
        # Map target mood to extended mood
        target_mood = None
        if target_mood_raw:
            target_mood = model_service.map_external_mood_to_extended(target_mood_raw)
        
        print(f"⏮️ Generating playlist from recently played tracks")
        
        # Get recently played from Spotify
        recent_tracks = await spotify_service.get_recently_played(
            access_token,
            limit=10
        )
        
        if not recent_tracks:
            raise HTTPException(status_code=404, detail="No recently played tracks found")
        
        print(f"✅ Got {len(recent_tracks)} recently played tracks")
        
        # Use most recent track as seed
        seed_track = recent_tracks[0]
        track_name = seed_track['name']
        artist_name = seed_track['artists'][0]['name']
        
        print(f"🌱 Using seed: {track_name} by {artist_name}")
        
        # Get recommendations
        recommendations = await music_service.get_recommendations(
            seed_track_name=track_name,
            seed_artist_name=artist_name,
            target_mood=target_mood,
            limit=limit * 2
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found")
        
        # Analyze with multi-mood support
        analyzed_tracks = []
        
        for track in recommendations:
            try:
                features = track.get('features')
                if not features:
                    continue
                
                lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                    track['name'],
                    track['artist']
                )
                
                mood_data = await model_service.predict_mood_from_features(
                    features,
                    lyrics_sentiment,
                    user_id=user_id
                )
                
                # Filter by target mood if specified
                if target_mood:
                    all_moods = mood_data.get('all_moods', [mood_data.get('primary_mood')])
                    if target_mood not in all_moods:
                        continue
                
                track['primary_mood'] = mood_data['primary_mood']
                track['all_moods'] = mood_data.get('all_moods', [])
                track['mood_scores'] = mood_data.get('mood_scores', {})
                track['confidence'] = mood_data['confidence']
                track['mood_details'] = mood_data
                
                analyzed_tracks.append(track)
                
                if len(analyzed_tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error analyzing track: {e}")
                continue
        
        # Calculate mood distribution
        mood_distribution = model_service.calculate_playlist_mood_distribution(analyzed_tracks)
        
        return {
            "source": "recently_played",
            "seed_track": {
                "name": track_name,
                "artist": artist_name,
                "played_at": seed_track.get('played_at')
            },
            "tracks": analyzed_tracks,
            "total": len(analyzed_tracks),
            "requested_mood": target_mood_raw,
            "target_mood": target_mood,
            "mood_distribution": mood_distribution,
            "approach": "hybrid_multi_mood",
            "mood_system": "12_extended_moods"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating from recently played: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# DISCOVERY ENDPOINTS (MULTI-MOOD)
# ============================================

@router.post("/discover")
async def discover_tracks(request: Dict[str, Any]):
    
    try:
        artist_name = request.get('artist_name')
        user_id = request.get('user_id')
        limit = request.get('limit', 20)
        
        if not artist_name:
            raise HTTPException(status_code=400, detail="Artist name is required")
        
        print(f"🔍 Discovering tracks similar to {artist_name}")
        
        similar_artists = await music_service.get_similar_artists_lastfm(
            artist_name,
            limit=10
        )
        
        if not similar_artists:
            return {
                "message": "No similar artists found",
                "tracks": []
            }
        
        discovered_tracks = []
        
        for artist in similar_artists[:5]:
            try:
                search_query = f"{artist['name']} official"
                tracks = await music_service.search_tracks(search_query, limit=3)
                
                for track in tracks:
                    features = await music_service.get_audio_features(
                        track['name'],
                        track['artists'][0] if track['artists'] else artist['name']
                    )
                    
                    if features:
                        lyrics_sentiment = {"polarity": 0.0, "subjectivity": 0.0}
                        
                        mood_data = await model_service.predict_mood_from_features(
                            features,
                            lyrics_sentiment,
                            user_id=user_id
                        )
                        
                        track['features'] = features
                        track['primary_mood'] = mood_data['primary_mood']
                        track['all_moods'] = mood_data.get('all_moods', [])
                        track['mood_scores'] = mood_data.get('mood_scores', {})
                        track['confidence'] = mood_data['confidence']
                        track['similar_to_artist'] = artist_name
                        track['match_score'] = artist['match_score']
                        
                        discovered_tracks.append(track)
                
                if len(discovered_tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"⚠️ Error discovering from artist {artist['name']}: {e}")
                continue
        
        discovered_tracks.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        # Calculate mood distribution
        mood_distribution = model_service.calculate_playlist_mood_distribution(discovered_tracks[:limit])
        
        return {
            "seed_artist": artist_name,
            "discovered_tracks": discovered_tracks[:limit],
            "total": len(discovered_tracks),
            "similar_artists_explored": len(similar_artists),
            "mood_distribution": mood_distribution,
            "mood_system": "12_extended_moods"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error discovering tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/personalized")
async def generate_personalized_playlist(
    request: Dict[str, Any],
    authorization: str = Header(None)
):
    
    try:
        user_id = request.get('user_id')
        limit = request.get('limit', 30)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        print(f"🎯 Generating personalized playlist for user {user_id}")
        
        user_stats_key = f"user_stats:{user_id}"
        user_stats = await cache_service.get_from_cache(user_stats_key)
        
        if not user_stats or user_stats.get('feedback_count', 0) < 5:
            return {
                "message": "Not enough feedback data for personalization. Please provide more feedback on tracks.",
                "tracks": [],
                "feedback_count": user_stats.get('feedback_count', 0) if user_stats else 0,
                "min_required": 5
            }
        
        # Get favorite mood from user corrections (now supports 12 moods)
        mood_corrections = user_stats.get('mood_corrections', {})
        if mood_corrections:
            favorite_mood = max(mood_corrections, key=mood_corrections.get)
        else:
            favorite_mood = "Relaxed"  # Default extended mood
        
        # Map to extended mood if it's a base mood
        if favorite_mood in model_service.BASE_MOOD_CLASSES:
            # Convert base mood to extended mood
            base_to_extended = {
                "Happy": "Joyful",
                "Sad": "Melancholic",
                "Calm": "Relaxed",
                "Energetic": "Motivated"
            }
            favorite_mood = base_to_extended.get(favorite_mood, "Relaxed")
        
        print(f"📊 User's favorite mood: {favorite_mood}")
        
        mood_request = GeneratePlaylistRequest(
            target_mood=favorite_mood,
            user_id=user_id,
            limit=limit
        )
        
        result = await generate_mood_playlist(mood_request, authorization)
        
        result['personalized'] = True
        result['user_preferences'] = {
            "favorite_mood": favorite_mood,
            "mood_distribution": mood_corrections,
            "feedback_count": user_stats.get('feedback_count', 0)
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating personalized playlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    
    return {
        "status": "healthy",
        "service": "playlist_generation",
        "approach": "hybrid_multi_mood",
        "mood_system": {
            "total_moods": 12,
            "extended_moods": model_service.ALL_MOOD_LABELS,
            "base_moods": model_service.BASE_MOOD_CLASSES,
            "multi_tag_support": True,
            "tags_per_track": "2-3",
            "similarity_threshold": "70%"
        },
        "features": [
            "12 Extended Moods (Relaxed, Focused, Romantic, Excited, Angry, Chill, Melancholic, Dreamy, Motivated, Joyful, Ambient, Party)",
            "Multi-Tag Classification (2-3 moods per track)",
            "Mood-based generation (Last.fm)",
            "Activity-based generation (16 activities)",
            "Discovery engine (Last.fm similar artists)",
            "Personalized playlists (user feedback)",
            "Spotify top tracks integration",
            "Spotify recently played integration",
            "Flow optimization (Dynamic Programming)",
            "External mood mapping (ANY mood → 12 extended moods)",
            "Mood distribution analytics",
            "Feature-based similarity matching"
        ],
        "supported_extended_moods": model_service.ALL_MOOD_LABELS,
        "supported_base_moods": model_service.BASE_MOOD_CLASSES,
        "mood_mapping_enabled": True,
        "data_sources": {
            "recommendations": "Last.fm",
            "user_data": "Spotify API",
            "audio_features": "Spotify API + AcousticBrainz + MusicBrainz",
            "mood_prediction": "ONNX ML Model + Feature Similarity + Lyrics"
        },
        "supported_activities": [
            "study", "workout", "gym", "party", "sleep", "meditation",
            "work", "focus", "driving", "relax", "chill", "romantic",
            "angry", "sad", "happy", "energetic"
        ]
    }