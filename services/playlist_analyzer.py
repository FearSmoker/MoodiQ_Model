"""
Advanced Playlist Audio Feature Aggregation
============================================
Implements weighted averaging with statistical methods for accurate playlist-level features
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


class PlaylistAnalyzer:
    """
    Advanced playlist analyzer using weighted statistical methods
    """
    
    # Feature weights based on importance for mood detection
    FEATURE_WEIGHTS = {
        'valence': 1.2,      # Most important for mood
        'energy': 1.2,       # Most important for mood
        'danceability': 1.0,
        'acousticness': 0.9,
        'instrumentalness': 0.8,
        'speechiness': 0.7,
        'tempo': 1.1,
        'loudness': 1.0,
        'liveness': 0.6,
        'key': 0.5,
        'mode': 0.5,
        'time_signature': 0.3
    }
    
    @staticmethod
    def calculate_weighted_mean(values: List[float], weights: Optional[List[float]] = None) -> float:
        """
        Calculate weighted mean with outlier handling
        
        Args:
            values: List of feature values
            weights: Optional weights for each value
            
        Returns:
            Weighted mean value
        """
        if not values:
            return 0.5
        
        values_array = np.array(values)
        
        # Remove outliers using IQR method
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        mask = (values_array >= lower_bound) & (values_array <= upper_bound)
        filtered_values = values_array[mask]
        
        if len(filtered_values) == 0:
            filtered_values = values_array  # Use all if all are outliers
        
        # Apply weights
        if weights:
            filtered_weights = np.array(weights)[mask] if len(mask) == len(weights) else np.ones(len(filtered_values))
            return float(np.average(filtered_values, weights=filtered_weights))
        
        return float(np.mean(filtered_values))
    
    @staticmethod
    def calculate_harmonic_mean(values: List[float]) -> float:
        """
        Calculate harmonic mean (useful for tempo-like features)
        """
        if not values or any(v <= 0 for v in values):
            return np.mean(values) if values else 0.5
        
        return float(len(values) / np.sum(1.0 / np.array(values)))
    
    @staticmethod
    def calculate_mode_distribution(values: List[int]) -> int:
        """
        Calculate mode (most frequent value) for categorical features
        """
        if not values:
            return 0
        
        from collections import Counter
        counter = Counter(values)
        return counter.most_common(1)[0][0]
    
    def aggregate_playlist_features(
        self,
        tracks: List[Dict],
        use_popularity_weighting: bool = True
    ) -> Dict[str, float]:
        """
        Advanced playlist feature aggregation with multiple statistical methods
        
        Args:
            tracks: List of track dictionaries with 'features' key
            use_popularity_weighting: Whether to weight by track popularity
            
        Returns:
            Dictionary of aggregated audio features
        """
        if not tracks:
            return self._get_default_features()
        
        # Extract all feature values
        feature_values = defaultdict(list)
        popularity_weights = []
        
        for track in tracks:
            features = track.get('features', {})
            if not features:
                continue
            
            # Extract each feature
            for feature_name in self.FEATURE_WEIGHTS.keys():
                value = features.get(feature_name)
                if value is not None:
                    feature_values[feature_name].append(float(value))
            
            # Get popularity for weighting
            if use_popularity_weighting:
                popularity = track.get('popularity', 50)
                popularity_weights.append(popularity / 100.0)  # Normalize to 0-1
        
        # Aggregate each feature using appropriate method
        aggregated = {}
        
        for feature_name, values in feature_values.items():
            if not values:
                aggregated[feature_name] = 0.5
                continue
            
            # Different aggregation methods for different feature types
            if feature_name == 'tempo':
                # Harmonic mean for tempo (better for rhythm)
                aggregated[feature_name] = self.calculate_harmonic_mean(values)
            
            elif feature_name in ['key', 'mode', 'time_signature']:
                # Mode (most frequent) for categorical features
                aggregated[feature_name] = float(self.calculate_mode_distribution([int(v) for v in values]))
            
            else:
                # Weighted mean for continuous features
                weights = None
                if use_popularity_weighting and len(popularity_weights) == len(values):
                    # Combine feature weight with popularity weight
                    feature_weight = self.FEATURE_WEIGHTS.get(feature_name, 1.0)
                    weights = [feature_weight * pop_weight for pop_weight in popularity_weights[:len(values)]]
                
                aggregated[feature_name] = self.calculate_weighted_mean(values, weights)
        
        # Add metadata
        aggregated['_metadata'] = {
            'track_count': len(tracks),
            'tracks_with_features': len([t for t in tracks if t.get('features')]),
            'aggregation_method': 'weighted_statistical',
            'popularity_weighted': use_popularity_weighting
        }
        
        return aggregated
    
    def calculate_playlist_diversity(self, tracks: List[Dict]) -> Dict[str, float]:
        """
        Calculate diversity metrics for playlist
        
        Returns:
            Dictionary with diversity scores
        """
        if not tracks:
            return {'overall_diversity': 0.0}
        
        feature_values = defaultdict(list)
        
        for track in tracks:
            features = track.get('features', {})
            for feature_name in ['valence', 'energy', 'danceability']:
                value = features.get(feature_name)
                if value is not None:
                    feature_values[feature_name].append(float(value))
        
        diversity_scores = {}
        
        for feature_name, values in feature_values.items():
            if values:
                # Standard deviation as diversity measure
                diversity_scores[f'{feature_name}_diversity'] = float(np.std(values))
        
        # Overall diversity (average of individual diversities)
        if diversity_scores:
            diversity_scores['overall_diversity'] = float(np.mean(list(diversity_scores.values())))
        else:
            diversity_scores['overall_diversity'] = 0.0
        
        return diversity_scores
    
    def calculate_energy_progression(self, tracks: List[Dict]) -> Dict[str, any]:
        """
        Analyze how energy progresses through playlist
        
        Returns:
            Dictionary with progression analysis
        """
        energy_values = []
        
        for track in tracks:
            features = track.get('features', {})
            energy = features.get('energy')
            if energy is not None:
                energy_values.append(float(energy))
        
        if not energy_values:
            return {'progression_type': 'unknown', 'trend': 0.0}
        
        # Calculate trend using linear regression
        x = np.arange(len(energy_values))
        coefficients = np.polyfit(x, energy_values, 1)
        trend = coefficients[0]  # Slope
        
        # Classify progression type
        if abs(trend) < 0.01:
            progression_type = 'stable'
        elif trend > 0:
            progression_type = 'building' if trend > 0.05 else 'gradually_building'
        else:
            progression_type = 'cooling_down' if trend < -0.05 else 'gradually_cooling'
        
        return {
            'progression_type': progression_type,
            'trend': float(trend),
            'start_energy': float(energy_values[0]),
            'end_energy': float(energy_values[-1]),
            'peak_energy': float(max(energy_values)),
            'peak_position': int(energy_values.index(max(energy_values))) / len(energy_values)
        }
    
    @staticmethod
    def _get_default_features() -> Dict[str, float]:
        """Return default features when no data available"""
        return {
            'valence': 0.5,
            'energy': 0.5,
            'danceability': 0.5,
            'acousticness': 0.5,
            'instrumentalness': 0.3,
            'speechiness': 0.1,
            'tempo': 120.0,
            'loudness': -10.0,
            'liveness': 0.1,
            'key': 0,
            'mode': 1,
            'time_signature': 4,
            '_metadata': {
                'track_count': 0,
                'tracks_with_features': 0,
                'aggregation_method': 'default'
            }
        }


# Global instance
playlist_analyzer = PlaylistAnalyzer()


def aggregate_playlist_features(tracks: List[Dict]) -> Dict[str, float]:
    """
    Convenience function for aggregating playlist features
    
    Args:
        tracks: List of track dictionaries
        
    Returns:
        Aggregated audio features
    """
    return playlist_analyzer.aggregate_playlist_features(tracks)


def analyze_playlist_comprehensive(tracks: List[Dict]) -> Dict:
    """
    Comprehensive playlist analysis including features, diversity, and progression
    
    Args:
        tracks: List of track dictionaries
        
    Returns:
        Complete analysis dictionary
    """
    return {
        'aggregated_features': playlist_analyzer.aggregate_playlist_features(tracks),
        'diversity_metrics': playlist_analyzer.calculate_playlist_diversity(tracks),
        'energy_progression': playlist_analyzer.calculate_energy_progression(tracks),
        'track_count': len(tracks)
    }