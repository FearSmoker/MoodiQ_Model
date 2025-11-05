"""
MoodiQ-AI Mood Classification Model Training Pipeline

This script trains a neural network to classify songs into moods based on 
Spotify audio features and exports the model to ONNX format.

Usage:
    python train_mood_model.py --dataset path/to/dataset.csv --output models/mood_model.onnx
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tf2onnx
import argparse
import os
import json
from datetime import datetime


# Configuration
# This will be overridden dynamically after loading the dataset
MOOD_CLASSES = []


AUDIO_FEATURES = [
    'valence', 'energy', 'danceability', 'acousticness',
    'instrumentalness', 'speechiness', 'tempo', 'loudness',
    'liveness', 'key', 'mode', 'time_signature'
]

# Model hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 15


def create_sample_dataset(n_samples=10000, save_path='sample_dataset.csv'):
    """
    Create a synthetic dataset for training if you don't have real labeled data.
    Replace this with your actual labeled dataset.
    
    Dataset format:
        - Columns: valence, energy, danceability, ..., mood (label)
        - mood: One of ["Happy", "Sad", "Calm", "Energetic", "Angry", "Focus"]
    """
    print(f"Creating sample dataset with {n_samples} samples...")
    
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        # Generate features with correlations to moods
        mood = np.random.choice(MOOD_CLASSES)
        
        if mood == "Happy":
            valence = np.random.uniform(0.6, 1.0)
            energy = np.random.uniform(0.6, 1.0)
            danceability = np.random.uniform(0.5, 1.0)
            acousticness = np.random.uniform(0.0, 0.4)
            tempo = np.random.uniform(120, 180)
            
        elif mood == "Sad":
            valence = np.random.uniform(0.0, 0.4)
            energy = np.random.uniform(0.0, 0.4)
            danceability = np.random.uniform(0.0, 0.5)
            acousticness = np.random.uniform(0.4, 1.0)
            tempo = np.random.uniform(60, 110)
            
        elif mood == "Calm":
            valence = np.random.uniform(0.4, 0.7)
            energy = np.random.uniform(0.0, 0.4)
            danceability = np.random.uniform(0.0, 0.5)
            acousticness = np.random.uniform(0.5, 1.0)
            tempo = np.random.uniform(70, 120)
            
        elif mood == "Energetic":
            valence = np.random.uniform(0.5, 1.0)
            energy = np.random.uniform(0.7, 1.0)
            danceability = np.random.uniform(0.6, 1.0)
            acousticness = np.random.uniform(0.0, 0.3)
            tempo = np.random.uniform(130, 200)
            
        elif mood == "Angry":
            valence = np.random.uniform(0.0, 0.4)
            energy = np.random.uniform(0.7, 1.0)
            danceability = np.random.uniform(0.3, 0.7)
            acousticness = np.random.uniform(0.0, 0.3)
            tempo = np.random.uniform(120, 180)
            
        else:  # Focus
            valence = np.random.uniform(0.4, 0.7)
            energy = np.random.uniform(0.4, 0.7)
            danceability = np.random.uniform(0.3, 0.6)
            acousticness = np.random.uniform(0.2, 0.6)
            tempo = np.random.uniform(90, 130)
        
        # Add noise
        valence = np.clip(valence + np.random.normal(0, 0.1), 0, 1)
        energy = np.clip(energy + np.random.normal(0, 0.1), 0, 1)
        danceability = np.clip(danceability + np.random.normal(0, 0.1), 0, 1)
        acousticness = np.clip(acousticness + np.random.normal(0, 0.1), 0, 1)
        
        # Generate other features
        instrumentalness = np.random.uniform(0, 0.8)
        speechiness = np.random.uniform(0, 0.3)
        loudness = np.random.uniform(-30, 0)
        liveness = np.random.uniform(0, 0.5)
        key = np.random.randint(0, 12)
        mode = np.random.randint(0, 2)
        time_signature = np.random.choice([3, 4, 5])
        
        data.append({
            'valence': valence,
            'energy': energy,
            'danceability': danceability,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'speechiness': speechiness,
            'tempo': tempo,
            'loudness': loudness,
            'liveness': liveness,
            'key': key,
            'mode': mode,
            'time_signature': time_signature,
            'mood': mood
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"✅ Sample dataset saved to {save_path}")
    
    return df


def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the dataset.
    
    Args:
        dataset_path: Path to CSV file with columns: audio features + 'mood'
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    print(f"Loading dataset from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found. Creating sample dataset...")
        df = create_sample_dataset(save_path=dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nMood distribution:")
    print(df['mood'].value_counts())

    # Dynamically detect mood classes from dataset
    global MOOD_CLASSES
    MOOD_CLASSES = sorted(df['mood'].unique())
    print(f"\n✅ Detected mood classes: {MOOD_CLASSES}")
    
    # Separate features and labels
    X = df[AUDIO_FEATURES].values
    y = df['mood'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(MOOD_CLASSES)
    y_encoded = label_encoder.transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(MOOD_CLASSES))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(MOOD_CLASSES))
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder


def build_model(input_dim, num_classes):
    """
    Build the neural network architecture.
    
    Architecture:
        - Input layer (12 features)
        - Dense layer (128 units, ReLU)
        - Dropout (0.3)
        - Dense layer (64 units, ReLU)
        - Dropout (0.3)
        - Dense layer (32 units, ReLU)
        - Dropout (0.2)
        - Output layer (6 classes, Softmax)
    """
    print("\nBuilding model architecture...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # Hidden layers with batch normalization
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        
        layers.Dense(32, activation='relu', name='dense_3'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    print(model.summary())
    
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the model with callbacks.
    """
    print("\nTraining model...")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance and create visualizations.
    """
    print("\nEvaluating model...")
    
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Confusion matrix saved to confusion_matrix.png")
    
    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for mood, acc in zip(label_encoder.classes_, class_accuracy):
        print(f"  {mood}: {acc:.3f}")
    
    return y_pred, y_pred_probs


def plot_training_history(history):
    """
    Plot training history.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✅ Training history saved to training_history.png")


def export_to_onnx(model, output_path, input_shape):
    """
    Export trained Keras model to ONNX format.
    """
    print(f"\nExporting model to ONNX format...")
    
    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to ONNX
    spec = (tf.TensorSpec((None, input_shape), tf.float32, name="input"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=output_path
    )
    
    print(f"✅ Model exported to {output_path}")
    print(f"   Input shape: (batch_size, {input_shape})")
    print(f"   Output shape: (batch_size, {len(MOOD_CLASSES)})")
    
    return output_path


def save_metadata(scaler, label_encoder, output_dir='models'):
    """
    Save preprocessing metadata for inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        'mood_classes': MOOD_CLASSES,
        'audio_features': AUDIO_FEATURES,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'timestamp': datetime.now().isoformat(),
        'model_version': '1.0.0'
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to {metadata_path}")


def main():
    """
    Main training pipeline.
    """
    parser = argparse.ArgumentParser(description='Train MoodiQ-AI mood classification model')
    parser.add_argument('--dataset', type=str, default='mood_dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--output', type=str, default='models/mood_model.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples to generate if dataset does not exist')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎵 MoodiQ-AI Mood Classification Model Training")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, label_encoder = load_and_preprocess_data(
        args.dataset
    )
    
    # Build model
    model = build_model(
        input_dim=len(AUDIO_FEATURES),
        num_classes=len(MOOD_CLASSES)
    )
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Plot training history
    plot_training_history(history)
    
    # Export to ONNX
    export_to_onnx(model, args.output, len(AUDIO_FEATURES))
    
    # Save metadata
    save_metadata(scaler, label_encoder)
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print("=" * 60)
    print(f"\nModel files created:")
    print(f"  - {args.output}")
    print(f"  - models/model_metadata.json")
    print(f"  - confusion_matrix.png")
    print(f"  - training_history.png")
    print(f"\nNext steps:")
    print(f"  1. Copy {args.output} to your ML service models/ directory")
    print(f"  2. Start your ML service: uvicorn main:app --reload --port 8000")
    print(f"  3. Test predictions: curl http://localhost:8000/predict/track")


if __name__ == '__main__':
    main()
