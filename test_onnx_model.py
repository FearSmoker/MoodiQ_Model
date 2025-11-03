"""
Test the trained ONNX model for mood prediction.

Usage:
    python test_onnx_model.py --model models/mood_model.onnx
"""

import numpy as np
import onnxruntime as ort
import json
import argparse
import os


# Updated to match the 4 moods from 278k_song_labelled.csv
MOOD_CLASSES = ["Calm", "Energetic", "Happy", "Sad"]

# Test cases with known moods (updated to use only 4 moods)
TEST_TRACKS = {
    "Happy Upbeat Pop": {
        "valence": 0.85,
        "energy": 0.80,
        "danceability": 0.75,
        "acousticness": 0.15,
        "instrumentalness": 0.0,
        "speechiness": 0.05,
        "tempo": 130.0,
        "loudness": -5.0,
        "liveness": 0.10,
        "key": 7,
        "mode": 1,
        "time_signature": 4,
        "expected_mood": "Happy"
    },
    "Sad Ballad": {
        "valence": 0.20,
        "energy": 0.30,
        "danceability": 0.40,
        "acousticness": 0.80,
        "instrumentalness": 0.0,
        "speechiness": 0.03,
        "tempo": 75.0,
        "loudness": -8.0,
        "liveness": 0.08,
        "key": 2,
        "mode": 0,
        "time_signature": 4,
        "expected_mood": "Sad"
    },
    "Calm Ambient": {
        "valence": 0.50,
        "energy": 0.25,
        "danceability": 0.35,
        "acousticness": 0.70,
        "instrumentalness": 0.90,
        "speechiness": 0.0,
        "tempo": 90.0,
        "loudness": -12.0,
        "liveness": 0.05,
        "key": 5,
        "mode": 1,
        "time_signature": 4,
        "expected_mood": "Calm"
    },
    "Energetic Dance": {
        "valence": 0.70,
        "energy": 0.95,
        "danceability": 0.90,
        "acousticness": 0.05,
        "instrumentalness": 0.40,
        "speechiness": 0.08,
        "tempo": 128.0,
        "loudness": -3.0,
        "liveness": 0.15,
        "key": 9,
        "mode": 1,
        "time_signature": 4,
        "expected_mood": "Energetic"
    },
    "Energetic Rock": {
        "valence": 0.65,
        "energy": 0.92,
        "danceability": 0.55,
        "acousticness": 0.08,
        "instrumentalness": 0.30,
        "speechiness": 0.10,
        "tempo": 145.0,
        "loudness": -4.0,
        "liveness": 0.18,
        "key": 0,
        "mode": 0,
        "time_signature": 4,
        "expected_mood": "Energetic"
    },
    "Calm Piano": {
        "valence": 0.45,
        "energy": 0.20,
        "danceability": 0.30,
        "acousticness": 0.85,
        "instrumentalness": 0.95,
        "speechiness": 0.0,
        "tempo": 85.0,
        "loudness": -15.0,
        "liveness": 0.05,
        "key": 4,
        "mode": 1,
        "time_signature": 4,
        "expected_mood": "Calm"
    }
}


def load_metadata(metadata_path='models/model_metadata.json'):
    """
    Load model metadata (scaler parameters, feature order, etc.)
    """
    if not os.path.exists(metadata_path):
        print(f"⚠️  Metadata file not found at {metadata_path}")
        print("   Using default normalization (no scaling)")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def normalize_features(features, metadata):
    """
    Normalize features using saved scaler parameters.
    """
    if metadata is None:
        # No normalization
        return features
    
    feature_order = metadata['audio_features']
    scaler_mean = np.array(metadata['scaler_mean'])
    scaler_scale = np.array(metadata['scaler_scale'])
    
    # Ensure features are in correct order
    normalized = []
    for feature_name in feature_order:
        value = features.get(feature_name, 0.5)
        normalized.append(value)
    
    normalized = np.array(normalized)
    
    # Apply standardization: (x - mean) / std
    normalized = (normalized - scaler_mean) / scaler_scale
    
    return normalized


def test_model(model_path, metadata_path='models/model_metadata.json'):
    """
    Test the ONNX model with predefined test cases.
    """
    print("=" * 60)
    print("🧪 Testing ONNX Model")
    print("=" * 60)
    
    # Load metadata
    metadata = load_metadata(metadata_path)
    if metadata:
        print(f"✅ Loaded metadata from {metadata_path}")
        # Verify mood classes match
        if 'mood_classes' in metadata:
            saved_classes = metadata['mood_classes']
            print(f"   Model trained on moods: {saved_classes}")
            if saved_classes != MOOD_CLASSES:
                print(f"⚠️  WARNING: Mood class mismatch!")
                print(f"   Expected: {MOOD_CLASSES}")
                print(f"   Found: {saved_classes}")
    
    # Load ONNX model
    print(f"\n📥 Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        print("\nPlease train the model first:")
        print("  python train_mood_model.py")
        return
    
    try:
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get model info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"\n📊 Model Information:")
    print(f"   Input name: {input_info.name}")
    print(f"   Input shape: {input_info.shape}")
    print(f"   Output name: {output_info.name}")
    print(f"   Output shape: {output_info.shape}")
    print(f"   Expected classes: {MOOD_CLASSES}")
    
    # Run tests
    print("\n" + "=" * 60)
    print("🎵 Running Test Cases")
    print("=" * 60)
    
    correct = 0
    total = len(TEST_TRACKS)
    
    for track_name, features in TEST_TRACKS.items():
        expected_mood = features.pop('expected_mood')
        
        # Normalize features
        if metadata:
            input_data = normalize_features(features, metadata)
        else:
            # Manual normalization for tempo and loudness
            input_data = [
                features['valence'],
                features['energy'],
                features['danceability'],
                features['acousticness'],
                features['instrumentalness'],
                features['speechiness'],
                min(features['tempo'] / 200.0, 1.0),  # Normalize tempo
                (features['loudness'] + 60) / 60.0,   # Normalize loudness
                features['liveness'],
                features['key'] / 11.0,  # Normalize key
                features['mode'],
                features['time_signature'] / 7.0  # Normalize time signature
            ]
            input_data = np.array(input_data, dtype=np.float32)
        
        # Reshape for batch prediction
        input_data = input_data.reshape(1, -1).astype(np.float32)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        
        # Get prediction
        probabilities = outputs[0][0]
        predicted_index = np.argmax(probabilities)
        predicted_mood = MOOD_CLASSES[predicted_index]
        confidence = probabilities[predicted_index]
        
        # Check if correct
        is_correct = predicted_mood == expected_mood
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\n{status} {track_name}")
        print(f"   Expected: {expected_mood}")
        print(f"   Predicted: {predicted_mood} ({confidence:.2%} confidence)")
        print(f"   All probabilities:")
        for mood, prob in zip(MOOD_CLASSES, probabilities):
            bar = "█" * int(prob * 20)
            print(f"      {mood:12s}: {prob:.2%} {bar}")
    
    # Summary
    accuracy = correct / total
    print("\n" + "=" * 60)
    print("📈 Test Results")
    print("=" * 60)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    
    if accuracy >= 0.8:
        print("✅ Model performance is good!")
    elif accuracy >= 0.5:
        print("⚠️  Model performance is acceptable but could be improved")
        print("   Consider training with more data or tuning hyperparameters")
    else:
        print("❌ Model performance is poor")
        print("   Consider collecting more training data or revising model architecture")
    
    return accuracy


def interactive_test(model_path, metadata_path='models/model_metadata.json'):
    """
    Interactive testing mode where user can input custom audio features.
    """
    print("\n" + "=" * 60)
    print("🎹 Interactive Test Mode")
    print("=" * 60)
    print(f"\nThis model predicts: {', '.join(MOOD_CLASSES)}")
    print("\nEnter audio features to predict mood:")
    
    # Load model
    metadata = load_metadata(metadata_path)
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    while True:
        try:
            print("\nEnter features (or 'quit' to exit):")
            
            features = {}
            features['valence'] = float(input("  Valence (0-1): "))
            features['energy'] = float(input("  Energy (0-1): "))
            features['danceability'] = float(input("  Danceability (0-1): "))
            features['acousticness'] = float(input("  Acousticness (0-1): "))
            features['instrumentalness'] = float(input("  Instrumentalness (0-1): "))
            features['speechiness'] = float(input("  Speechiness (0-1): "))
            features['tempo'] = float(input("  Tempo (BPM): "))
            features['loudness'] = float(input("  Loudness (dB): "))
            features['liveness'] = float(input("  Liveness (0-1): "))
            features['key'] = int(input("  Key (0-11): "))
            features['mode'] = int(input("  Mode (0-1): "))
            features['time_signature'] = int(input("  Time Signature (3-7): "))
            
            # Normalize and predict
            if metadata:
                input_data = normalize_features(features, metadata)
            else:
                input_data = [
                    features['valence'], features['energy'], features['danceability'],
                    features['acousticness'], features['instrumentalness'], features['speechiness'],
                    min(features['tempo'] / 200.0, 1.0), (features['loudness'] + 60) / 60.0,
                    features['liveness'], features['key'] / 11.0, features['mode'],
                    features['time_signature'] / 7.0
                ]
                input_data = np.array(input_data, dtype=np.float32)
            
            input_data = input_data.reshape(1, -1).astype(np.float32)
            
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data})
            
            probabilities = outputs[0][0]
            predicted_index = np.argmax(probabilities)
            predicted_mood = MOOD_CLASSES[predicted_index]
            
            print(f"\n🎯 Predicted Mood: {predicted_mood}")
            print(f"   Probabilities:")
            for mood, prob in zip(MOOD_CLASSES, probabilities):
                bar = "█" * int(prob * 20)
                print(f"      {mood:12s}: {prob:.2%} {bar}")
            
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Exiting interactive mode")
            break


def main():
    """
    Main testing function.
    """
    parser = argparse.ArgumentParser(description='Test ONNX mood prediction model')
    parser.add_argument('--model', type=str, default='models/mood_model.onnx',
                        help='Path to ONNX model file')
    parser.add_argument('--metadata', type=str, default='models/model_metadata.json',
                        help='Path to model metadata JSON')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test(args.model, args.metadata)
    else:
        test_model(args.model, args.metadata)


if __name__ == '__main__':
    main()