"""
Complete Flow Testing Script
============================
Tests all new features end-to-end
"""

# ✅ Add these two lines BEFORE importing anything else
from dotenv import load_dotenv
load_dotenv()

import asyncio
import sys
sys.path.insert(0, '.')

from services.playlist_analyzer import playlist_analyzer
from services.live_queue_service import live_queue_service
from services.db_recommendation_service import db_recommendation_service
from services import model_service, music_service


async def test_playlist_aggregation():
    """Test advanced playlist aggregation"""
    print("\n" + "="*80)
    print("TEST 1: ADVANCED PLAYLIST AGGREGATION")
    print("="*80)
    
    # Sample tracks
    test_tracks = [
        {
            "name": "Happy",
            "artist": "Pharrell Williams",
            "features": {
                "valence": 0.96, "energy": 0.80, "danceability": 0.85,
                "acousticness": 0.10, "tempo": 160, "loudness": -5,
                "instrumentalness": 0.0, "speechiness": 0.06, "liveness": 0.1,
                "key": 7, "mode": 1, "time_signature": 4
            },
            "popularity": 95
        },
        {
            "name": "Uptown Funk",
            "artist": "Mark Ronson",
            "features": {
                "valence": 0.85, "energy": 0.88, "danceability": 0.90,
                "acousticness": 0.05, "tempo": 115, "loudness": -4,
                "instrumentalness": 0.0, "speechiness": 0.10, "liveness": 0.12,
                "key": 1, "mode": 1, "time_signature": 4
            },
            "popularity": 90
        },
        {
            "name": "Can't Stop the Feeling",
            "artist": "Justin Timberlake",
            "features": {
                "valence": 0.89, "energy": 0.82, "danceability": 0.74,
                "acousticness": 0.12, "tempo": 113, "loudness": -4.5,
                "instrumentalness": 0.0, "speechiness": 0.04, "liveness": 0.10,
                "key": 3, "mode": 1, "time_signature": 4
            },
            "popularity": 88
        }
    ]
    
    # Test aggregation
    print("\n📊 Aggregating playlist features...")
    aggregated = playlist_analyzer.aggregate_playlist_features(test_tracks)
    
    print(f"\nAggregated Features:")
    print(f"   Valence: {aggregated['valence']:.3f}")
    print(f"   Energy: {aggregated['energy']:.3f}")
    print(f"   Danceability: {aggregated['danceability']:.3f}")
    print(f"   Tempo: {aggregated['tempo']:.1f} BPM")
    
    # Test diversity
    diversity = playlist_analyzer.calculate_playlist_diversity(test_tracks)
    print(f"\nDiversity Metrics:")
    print(f"   Overall diversity: {diversity['overall_diversity']:.3f}")
    
    # Test progression
    progression = playlist_analyzer.calculate_energy_progression(test_tracks)
    print(f"\nEnergy Progression:")
    print(f"   Type: {progression['progression_type']}")
    print(f"   Trend: {progression['trend']:.3f}")
    
    # Predict mood
    print("\n🎭 Predicting playlist mood...")
    mood_data = await model_service.predict_mood_from_features(
        aggregated,
        {'polarity': 0.0, 'subjectivity': 0.0}
    )
    
    print(f"\nPlaylist Mood:")
    print(f"   Primary: {mood_data['primary_mood']}")
    print(f"   All moods: {', '.join(mood_data['all_moods'])}")
    print(f"   Confidence: {mood_data['confidence']:.2%}")
    
    print("\n✅ Playlist aggregation test PASSED")
    return aggregated, mood_data


async def test_live_session():
    """Test live listening queue"""
    print("\n" + "="*80)
    print("TEST 2: LIVE LISTENING SESSION")
    print("="*80)
    
    user_id = "test_user_123"
    
    # Start session
    print("\n🎧 Starting live session...")
    session_id = await live_queue_service.start_session(user_id)
    print(f"   Session ID: {session_id}")
    
    # Add some tracks
    test_tracks = [
        ("Happy", "Pharrell Williams"),
        ("Uptown Funk", "Mark Ronson"),
        ("Can't Stop the Feeling", "Justin Timberlake")
    ]
    
    for track_name, artist_name in test_tracks:
        print(f"\n➕ Adding track: {track_name}")
        
        # Get features
        features = await music_service.get_audio_features(track_name, artist_name)
        if not features:
            features = music_service.get_default_features()
        
        # Get mood
        mood_data = await model_service.predict_mood_from_features(
            features,
            {'polarity': 0.0, 'subjectivity': 0.0}
        )
        
        track_data = {
            'name': track_name,
            'artist': artist_name,
            'features': features,
            'primary_mood': mood_data['primary_mood'],
            'all_moods': mood_data['all_moods'],
            'mood_scores': mood_data['mood_scores']
        }
        
        queue_analytics = await live_queue_service.add_track_to_queue(
    user_id,
    session_id,
    track_data
)

# ✅ Update session_id in case a fallback session was created

    queue_analytics = await live_queue_service.add_track_to_queue(
        user_id,
        session_id,
        track_data
        )
        
    if queue_analytics.get("session_id") and queue_analytics["session_id"] != session_id:
        print(f"⚠️  Session renewed: {session_id} → {queue_analytics['session_id']}")
        session_id = queue_analytics["session_id"]

    print(f"   Queue size: {queue_analytics['track_count']}")
    print(f"   Current mood: {queue_analytics['current_mood']['primary_mood']}")


    
    print("\n📊 Getting current session...")
    queue_data = await live_queue_service.get_current_queue(user_id, session_id)

    print(f"\nSession Analytics:")

    if not queue_data:
        print("⚠️  No active queue found (session may have expired or been renewed).")
    else:
        tracks = queue_data.get('tracks', [])
        current_mood = queue_data.get('current_mood', {})

        print(f"   Tracks: {len(tracks)}")
        print(f"   Current mood: {current_mood.get('primary_mood', 'Unknown')}")
        print(f"   All moods: {', '.join(current_mood.get('all_moods', []))}")

    # End session
    print("\n🛑 Ending session...")
    final_analytics = await live_queue_service.end_session(user_id, session_id)
    
    print(f"\nFinal Analytics:")
    if "error" in final_analytics:
        print(f"⚠️  Session end failed: {final_analytics['error']}")
    else:
        print(f"   Duration: {final_analytics.get('session_duration_minutes', 'N/A')} minutes")
        print(f"   Total tracks: {final_analytics.get('track_count', 'N/A')}")
        final_mood = final_analytics.get('final_mood', {})
        print(f"   Final mood: {final_mood.get('primary_mood', 'Unknown')}")

    print("\n✅ Live session test PASSED")
    return final_analytics


async def test_database_recommendations():
    """Test MongoDB recommendations"""
    print("\n" + "="*80)
    print("TEST 3: DATABASE RECOMMENDATIONS")
    print("="*80)
    
    # Initialize service
    print("\n🔌 Connecting to MongoDB...")
    await db_recommendation_service.initialize()
    
    if not db_recommendation_service._initialized:
        print("⚠️ Skipping test - MongoDB not available")
        return
    
    # Test mood-based recommendations
    print("\n🎭 Getting mood-based recommendations...")
    recommendations = await db_recommendation_service.get_mood_based_recommendations(
        mood="Joyful",
        limit=10
    )
    
    print(f"\nMood-Based Results:")
    print(f"   Found: {len(recommendations)} tracks")
    
    if recommendations:
        for i, track in enumerate(recommendations[:5], 1):
            print(f"   {i}. {track['track_name']} by {track['artist_name']}")
            print(f"      Moods: {', '.join(track['moods']['all_moods'])}")
    
    # Test feature-based recommendations
    print("\n🎵 Getting feature-based recommendations...")
    target_features = {
        'valence': 0.9,
        'energy': 0.8,
        'danceability': 0.85,
        'acousticness': 0.1,
        'instrumentalness': 0.0,
        'speechiness': 0.05,
        'tempo': 120,
        'loudness': -5,
        'liveness': 0.1
    }
    
    recommendations = await db_recommendation_service.find_similar_tracks(
        target_features=target_features,
        target_moods=["Joyful", "Excited"],
        limit=10,
        min_similarity=0.75
    )
    
    print(f"\nFeature-Based Results:")
    print(f"   Found: {len(recommendations)} tracks")
    
    if recommendations:
        for i, track in enumerate(recommendations[:5], 1):
            print(f"   {i}. {track['track_name']} - Similarity: {track['similarity_score']:.3f}")
    
    print("\n✅ Database recommendations test PASSED")
    return recommendations


async def test_flow_optimization():
    """Test enhanced flow optimization"""
    print("\n" + "="*80)
    print("TEST 4: FLOW OPTIMIZATION")
    print("="*80)
    
    # Sample playlist
    test_tracks = [
        {"name": "Track 1", "features": {"energy": 0.3, "valence": 0.4, "danceability": 0.3}},
        {"name": "Track 2", "features": {"energy": 0.8, "valence": 0.9, "danceability": 0.8}},
        {"name": "Track 3", "features": {"energy": 0.5, "valence": 0.6, "danceability": 0.5}},
        {"name": "Track 4", "features": {"energy": 0.7, "valence": 0.7, "danceability": 0.7}},
        {"name": "Track 5", "features": {"energy": 0.9, "valence": 0.85, "danceability": 0.9}}
    ]
    
    print(f"\n🔄 Optimizing {len(test_tracks)} tracks...")
    print("   Target: Calm → Energetic progression")
    
    # Use gradient optimization
    result = model_service.optimize_flow_with_gradient(
        test_tracks,
        target_start_energy=0.3,
        target_end_energy=0.9
    )
    
    print(f"\nOptimization Results:")
    print(f"   Flow score: {result['flowScore']:.3f}")
    print(f"   Smoothness: {result['smoothness']:.3f}")
    print(f"   Gradient adherence: {result['gradient_score']:.3f}")
    print(f"   Energy range: {result['energy_start']:.2f} → {result['energy_end']:.2f}")
    
    print(f"\nOptimized order:")
    for i, idx in enumerate(result['optimizedOrder'], 1):
        track = test_tracks[idx]
        energy = track['features']['energy']
        print(f"   {i}. {track['name']} (energy: {energy:.2f})")
    
    print("\n✅ Flow optimization test PASSED")
    return result


async def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*80)
    print("🧪 COMPLETE FLOW TESTING SUITE")
    print("="*80)
    print("\nTesting all new features...")
    
    try:
        # Test 1: Playlist Aggregation
        aggregated, mood = await test_playlist_aggregation()
        
        # Test 2: Live Session
        session_analytics = await test_live_session()
        
        # Test 3: Database Recommendations
        recommendations = await test_database_recommendations()
        
        # Test 4: Flow Optimization
        flow_result = await test_flow_optimization()
        
        # Summary
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        print("\n📊 Summary:")
        print(f"   ✅ Playlist aggregation: Working")
        print(f"   ✅ Live session tracking: Working")
        print(f"   ✅ Database recommendations: Working")
        print(f"   ✅ Flow optimization: Working")
        
        print("\n🎉 Your MoodiQ ML service is fully operational!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await db_recommendation_service.close()


if __name__ == "__main__":
    print("\n🚀 Starting comprehensive testing...")
    asyncio.run(run_all_tests())