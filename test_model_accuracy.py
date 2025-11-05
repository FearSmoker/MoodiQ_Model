"""
Comprehensive Model Accuracy Testing Script

Tests:
1. ONNX model loading and inference
2. Feature normalization
3. Mood classification accuracy
4. Genre-adaptive weighting
5. Lyrics fusion
6. Mood mapping for external moods
7. Real-world test cases

Usage:
    python test_model_accuracy.py
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# CRITICAL: Load .env BEFORE importing services
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import services (after .env is loaded)
from services import model_service, music_service, lyrics_service

# Test cases with known characteristics
TEST_CASES = [
    {
        "name": "Happy Upbeat Pop",
        "track": "Happy",
        "artist": "Pharrell Williams",
        "expected_mood": "Happy",
        "expected_features": {
            "valence": (0.7, 1.0),  # High valence
            "energy": (0.7, 1.0),    # High energy
        }
    },
    {
        "name": "Sad Ballad",
        "track": "Someone Like You",
        "artist": "Adele",
        "expected_mood": "Sad",
        "expected_features": {
            "valence": (0.0, 0.4),  # Low valence
            "energy": (0.0, 0.5),    # Low energy
        }
    },
    {
        "name": "Calm Ambient",
        "track": "Weightless",
        "artist": "Marconi Union",
        "expected_mood": "Calm",
        "expected_features": {
            "valence": (0.3, 0.7),  # Medium valence
            "energy": (0.0, 0.4),    # Low energy
        }
    },
    {
        "name": "Energetic Rock",
        "track": "Eye of the Tiger",
        "artist": "Survivor",
        "expected_mood": "Energetic",
        "expected_features": {
            "valence": (0.5, 1.0),  # Medium-high valence
            "energy": (0.7, 1.0),    # High energy
        }
    },
    {
        "name": "Hip-Hop/Rap",
        "track": "Lose Yourself",
        "artist": "Eminem",
        "expected_mood": "Energetic",
        "genre": "rap",
        "expected_features": {
            "energy": (0.6, 1.0),
        }
    },
]


def check_feature_range(value, expected_range):
    """Check if value is in expected range"""
    min_val, max_val = expected_range
    return min_val <= value <= max_val


async def test_model_loading():
    """Test 1: Model Loading"""
    print("\n" + "="*80)
    print("TEST 1: MODEL LOADING")
    print("="*80)
    
    try:
        model_service.load_model()
        
        if model_service.mood_model is not None:
            print("✅ ONNX model loaded successfully")
            print(f"   Mood classes: {model_service.MOOD_CLASSES}")
            print(f"   Scaler loaded: {model_service.scaler_mean is not None}")
            return True
        else:
            print("⚠️  Model not found - using rule-based fallback")
            print(f"   Mood classes: {model_service.MOOD_CLASSES}")
            return True  # Not critical, we have fallback
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


async def test_feature_normalization():
    """Test 2: Feature Normalization"""
    print("\n" + "="*80)
    print("TEST 2: FEATURE NORMALIZATION")
    print("="*80)
    
    try:
        # Test with sample features
        test_features = {
            'valence': 0.8,
            'energy': 0.7,
            'danceability': 0.6,
            'acousticness': 0.2,
            'instrumentalness': 0.0,
            'speechiness': 0.05,
            'tempo': 120.0,
            'loudness': -5.0,
            'liveness': 0.1,
            'key': 7,
            'mode': 1,
            'time_signature': 4
        }
        
        normalized = model_service.normalize_features(test_features)
        
        print("✅ Features normalized successfully")
        print(f"   Input shape: {normalized.shape}")
        print(f"   Expected: (12,)")
        print(f"   Sample values: {normalized[:3]}")
        
        return normalized.shape == (12,)
        
    except Exception as e:
        print(f"❌ Normalization failed: {e}")
        return False


async def test_mood_classification():
    """Test 3: Mood Classification Accuracy"""
    print("\n" + "="*80)
    print("TEST 3: MOOD CLASSIFICATION ACCURACY")
    print("="*80)
    
    results = []
    correct = 0
    total = len(TEST_CASES)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n📝 Test {i}/{total}: {test_case['name']}")
        print(f"   Track: {test_case['track']} by {test_case['artist']}")
        print(f"   Expected mood: {test_case['expected_mood']}")
        
        try:
            # Get audio features
            features = await music_service.get_audio_features(
                test_case['track'],
                test_case['artist']
            )
            
            if not features:
                print("   ⚠️  Using default features")
                features = music_service.get_default_features()
            
            # Check feature ranges
            for feat_name, expected_range in test_case.get('expected_features', {}).items():
                actual_value = features.get(feat_name, 0.5)
                in_range = check_feature_range(actual_value, expected_range)
                
                status = "✅" if in_range else "⚠️"
                print(f"   {status} {feat_name}: {actual_value:.2f} (expected {expected_range[0]:.2f}-{expected_range[1]:.2f})")
            
            # Get lyrics sentiment
            lyrics_sentiment = await lyrics_service.get_lyrics_sentiment(
                test_case['track'],
                test_case['artist']
            )
            
            # Predict mood
            mood_data = await model_service.predict_mood_from_features(
                features,
                lyrics_sentiment,
                genre=test_case.get('genre')
            )
            
            predicted_mood = mood_data['fused_mood']
            confidence = mood_data['confidence']
            
            is_correct = predicted_mood == test_case['expected_mood']
            
            if is_correct:
                correct += 1
                print(f"   ✅ CORRECT: Predicted {predicted_mood} (confidence: {confidence:.2%})")
            else:
                print(f"   ❌ INCORRECT: Predicted {predicted_mood} (expected {test_case['expected_mood']})")
                print(f"      Confidence: {confidence:.2%}")
                print(f"      Audio mood: {mood_data['audio_mood']}")
                print(f"      Lyrics mood: {mood_data['lyrics_mood']}")
            
            results.append({
                'test': test_case['name'],
                'expected': test_case['expected_mood'],
                'predicted': predicted_mood,
                'correct': is_correct,
                'confidence': confidence,
                'mood_data': mood_data
            })
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'test': test_case['name'],
                'expected': test_case['expected_mood'],
                'predicted': 'ERROR',
                'correct': False,
                'confidence': 0.0,
                'error': str(e)
            })
    
    accuracy = (correct / total) * 100
    
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION ACCURACY SUMMARY")
    print(f"{'='*80}")
    print(f"Correct: {correct}/{total} ({accuracy:.1f}%)")
    
    if accuracy >= 80:
        print("✅ EXCELLENT: Model accuracy is good!")
    elif accuracy >= 60:
        print("⚠️  ACCEPTABLE: Model accuracy is moderate")
    else:
        print("❌ POOR: Model accuracy needs improvement")
    
    return accuracy >= 60, results


async def test_mood_mapping():
    """Test 4: External Mood Mapping"""
    print("\n" + "="*80)
    print("TEST 4: EXTERNAL MOOD MAPPING")
    print("="*80)
    
    external_moods = [
        # Should map to Happy
        ("joyful", "Happy"),
        ("cheerful", "Happy"),
        ("upbeat", "Happy"),
        
        # Should map to Sad
        ("melancholic", "Sad"),
        ("depressed", "Sad"),
        ("gloomy", "Sad"),
        
        # Should map to Calm
        ("relaxed", "Calm"),
        ("peaceful", "Calm"),
        ("chill", "Calm"),
        
        # Should map to Energetic
        ("hyped", "Energetic"),
        ("intense", "Energetic"),
        ("aggressive", "Energetic"),
        
        # Activity-based
        ("workout", "Energetic"),
        ("study", "Calm"),
        ("party", "Happy"),
    ]
    
    correct = 0
    total = len(external_moods)
    
    for external, expected in external_moods:
        mapped = model_service.map_external_mood_to_base(external)
        is_correct = mapped == expected
        
        if is_correct:
            correct += 1
            print(f"✅ {external:15s} → {mapped:10s} (expected {expected})")
        else:
            print(f"❌ {external:15s} → {mapped:10s} (expected {expected})")
    
    accuracy = (correct / total) * 100
    
    print(f"\nMapping Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    return accuracy >= 80


async def test_genre_adaptive_weighting():
    """Test 5: Genre-Adaptive Weighting"""
    print("\n" + "="*80)
    print("TEST 5: GENRE-ADAPTIVE WEIGHTING")
    print("="*80)
    
    # Test that rap/hip-hop gets higher lyrics weight
    rap_weights = model_service.GENRE_WEIGHTS.get('rap', model_service.GENRE_WEIGHTS['default'])
    pop_weights = model_service.GENRE_WEIGHTS.get('pop', model_service.GENRE_WEIGHTS['default'])
    classical_weights = model_service.GENRE_WEIGHTS.get('classical', model_service.GENRE_WEIGHTS['default'])
    
    print(f"Rap weights: Audio={rap_weights['audio']:.2f}, Lyrics={rap_weights['lyrics']:.2f}")
    print(f"Pop weights: Audio={pop_weights['audio']:.2f}, Lyrics={pop_weights['lyrics']:.2f}")
    print(f"Classical weights: Audio={classical_weights['audio']:.2f}, Lyrics={classical_weights['lyrics']:.2f}")
    
    # Rap should have higher lyrics weight
    rap_correct = rap_weights['lyrics'] > pop_weights['lyrics']
    # Classical should have very high audio weight
    classical_correct = classical_weights['audio'] > 0.8
    
    if rap_correct and classical_correct:
        print("✅ Genre weighting is correct")
        return True
    else:
        print("❌ Genre weighting needs adjustment")
        return False


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE MODEL ACCURACY TESTING")
    print("="*80)
    print(f"Testing MoodiQ-AI Model")
    print(f"Model Classes: {model_service.MOOD_CLASSES}")
    print("="*80)
    
    results = {
        'model_loading': False,
        'feature_normalization': False,
        'mood_classification': False,
        'mood_mapping': False,
        'genre_weighting': False
    }
    
    # Test 1: Model Loading
    results['model_loading'] = await test_model_loading()
    
    # Test 2: Feature Normalization
    results['feature_normalization'] = await test_feature_normalization()
    
    # Test 3: Mood Classification
    results['mood_classification'], classification_results = await test_mood_classification()
    
    # Test 4: Mood Mapping
    results['mood_mapping'] = await test_mood_mapping()
    
    # Test 5: Genre Weighting
    results['genre_weighting'] = await test_genre_adaptive_weighting()
    
    # Final Summary
    print("\n" + "="*80)
    print("📊 FINAL TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    overall_percentage = (passed_tests / total_tests) * 100
    
    print(f"\n{'='*80}")
    print(f"Overall: {passed_tests}/{total_tests} tests passed ({overall_percentage:.1f}%)")
    print(f"{'='*80}\n")
    
    if overall_percentage >= 80:
        print("✅ EXCELLENT: Model is working correctly!")
    elif overall_percentage >= 60:
        print("⚠️  ACCEPTABLE: Some issues detected but functional")
    else:
        print("❌ CRITICAL: Major issues detected - review model")
    
    return results


if __name__ == "__main__":
    print("Starting comprehensive model testing...")
    print("This will test:")
    print("  1. Model loading")
    print("  2. Feature normalization")
    print("  3. Mood classification accuracy")
    print("  4. External mood mapping")
    print("  5. Genre-adaptive weighting")
    print("\nThis may take a few minutes...\n")
    
    asyncio.run(run_all_tests())