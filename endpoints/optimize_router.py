from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from services import model_service

router = APIRouter()


class OptimizeFlowRequest(BaseModel):
    """
    Request model matching backend's POST /api/playlists/optimize
    """
    tracks: List[Dict[str, Any]]  # Track objects with mood and features
    start_mood: Optional[Dict[str, float]] = None
    end_mood: Optional[Dict[str, float]] = None
    algorithm: str = 'dynamic_programming'
    user_id: Optional[str] = None


class OptimizeFlowResponse(BaseModel):
    """Response matching backend expectations"""
    optimizedOrder: List[int]
    flowScore: float
    transitions: List[Dict[str, Any]]


@router.post("/flow", response_model=OptimizeFlowResponse)
async def optimize_playlist_flow(request: OptimizeFlowRequest):
    """
    Optimize playlist order for smooth mood transitions.
    
    This endpoint matches the backend's POST /api/playlists/optimize.
    Uses Dynamic Programming to find the optimal song order.
    """
    try:
        if not request.tracks or len(request.tracks) == 0:
            raise HTTPException(
                status_code=400,
                detail="Tracks array is required and cannot be empty"
            )
        
        print(f"🔄 Optimizing flow for {len(request.tracks)} tracks using {request.algorithm}")
        
        # Set default start/end moods if not provided
        if not request.start_mood:
            # Default: Start with calm energy
            request.start_mood = {
                'valence': 0.5,
                'energy': 0.4,
                'danceability': 0.5
            }
        
        if not request.end_mood:
            # Default: End with uplifting mood
            request.end_mood = {
                'valence': 0.7,
                'energy': 0.6,
                'danceability': 0.6
            }
        
        # Validate algorithm choice
        if request.algorithm not in ['dynamic_programming', 'greedy', 'simulated_annealing']:
            print(f"⚠️ Unknown algorithm '{request.algorithm}', using dynamic_programming")
            request.algorithm = 'dynamic_programming'
        
        # Run optimization
        if request.algorithm == 'dynamic_programming':
            result = model_service.optimize_flow_dp(
                request.tracks,
                request.start_mood,
                request.end_mood
            )
        elif request.algorithm == 'greedy':
            result = _optimize_flow_greedy(
                request.tracks,
                request.start_mood,
                request.end_mood
            )
        elif request.algorithm == 'simulated_annealing':
            result = _optimize_flow_simulated_annealing(
                request.tracks,
                request.start_mood,
                request.end_mood
            )
        else:
            # Default to DP
            result = model_service.optimize_flow_dp(
                request.tracks,
                request.start_mood,
                request.end_mood
            )
        
        print(f"✅ Flow optimization complete. Score: {result['flowScore']:.3f}")
        
        return {
            "optimizedOrder": result['optimizedOrder'],
            "flowScore": result['flowScore'],
            "transitions": result['transitions']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Flow optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize playlist flow: {str(e)}"
        )


def _optimize_flow_greedy(
    tracks: List[Dict], 
    start_mood: Dict, 
    end_mood: Dict
) -> Dict:
    """
    Simple greedy algorithm for playlist optimization.
    Always picks the nearest next track.
    Faster but less optimal than DP.
    """
    import numpy as np
    
    n = len(tracks)
    if n == 0:
        return {"optimizedOrder": [], "flowScore": 0, "transitions": []}
    
    def mood_distance(m1: Dict, m2: Dict) -> float:
        v1 = m1.get('valence', 0.5)
        e1 = m1.get('energy', 0.5)
        v2 = m2.get('valence', 0.5)
        e2 = m2.get('energy', 0.5)
        return np.sqrt((v1 - v2)**2 + (e1 - e2)**2)
    
    # Extract moods
    track_moods = []
    for track in tracks:
        if 'mood' in track and 'scores' in track['mood']:
            track_moods.append(track['mood']['scores'])
        elif 'features' in track:
            track_moods.append(track['features'])
        else:
            track_moods.append({'valence': 0.5, 'energy': 0.5})
    
    # Greedy selection
    used = [False] * n
    order = []
    current_mood = start_mood
    total_cost = 0
    
    # Start with track closest to start_mood
    distances = [mood_distance(start_mood, m) for m in track_moods]
    current_idx = np.argmin(distances)
    order.append(current_idx)
    used[current_idx] = True
    total_cost += distances[current_idx]
    current_mood = track_moods[current_idx]
    
    # Greedily add nearest tracks
    transitions = []
    for _ in range(n - 1):
        min_dist = float('inf')
        next_idx = -1
        
        for i in range(n):
            if not used[i]:
                dist = mood_distance(current_mood, track_moods[i])
                if dist < min_dist:
                    min_dist = dist
                    next_idx = i
        
        if next_idx != -1:
            transitions.append({
                "from_index": int(order[-1]),
                "to_index": int(next_idx),
                "smoothness": float(max(0, 1 - min_dist / 2)),
                "distance": float(min_dist)
            })
            
            order.append(next_idx)
            used[next_idx] = True
            total_cost += min_dist
            current_mood = track_moods[next_idx]
    
    # Add cost to end mood
    total_cost += mood_distance(current_mood, end_mood)
    
    # Calculate score
    max_possible_cost = n * 2.0
    flow_score = max(0, 1 - (total_cost / max_possible_cost))
    
    return {
        "optimizedOrder": order,
        "flowScore": float(flow_score),
        "transitions": transitions,
        "totalCost": float(total_cost)
    }


def _optimize_flow_simulated_annealing(
    tracks: List[Dict],
    start_mood: Dict,
    end_mood: Dict,
    max_iterations: int = 1000,
    initial_temp: float = 100.0
) -> Dict:
    """
    Simulated Annealing algorithm for playlist optimization.
    Can escape local minima for better global optimum.
    """
    import numpy as np
    import random
    
    n = len(tracks)
    if n == 0:
        return {"optimizedOrder": [], "flowScore": 0, "transitions": []}
    
    def mood_distance(m1: Dict, m2: Dict) -> float:
        v1 = m1.get('valence', 0.5)
        e1 = m1.get('energy', 0.5)
        v2 = m2.get('valence', 0.5)
        e2 = m2.get('energy', 0.5)
        return np.sqrt((v1 - v2)**2 + (e1 - e2)**2)
    
    def calculate_total_cost(order: List[int]) -> float:
        cost = 0
        # Cost from start to first track
        cost += mood_distance(start_mood, track_moods[order[0]])
        # Cost between consecutive tracks
        for i in range(len(order) - 1):
            cost += mood_distance(track_moods[order[i]], track_moods[order[i+1]])
        # Cost from last track to end
        cost += mood_distance(track_moods[order[-1]], end_mood)
        return cost
    
    # Extract moods
    track_moods = []
    for track in tracks:
        if 'mood' in track and 'scores' in track['mood']:
            track_moods.append(track['mood']['scores'])
        elif 'features' in track:
            track_moods.append(track['features'])
        else:
            track_moods.append({'valence': 0.5, 'energy': 0.5})
    
    # Initialize with random order
    current_order = list(range(n))
    random.shuffle(current_order)
    current_cost = calculate_total_cost(current_order)
    
    best_order = current_order.copy()
    best_cost = current_cost
    
    # Simulated annealing
    temp = initial_temp
    cooling_rate = 0.995
    
    for iteration in range(max_iterations):
        # Generate neighbor by swapping two random positions
        new_order = current_order.copy()
        i, j = random.sample(range(n), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_cost = calculate_total_cost(new_order)
        cost_diff = new_cost - current_cost
        
        # Accept if better, or with probability based on temperature
        if cost_diff < 0 or random.random() < np.exp(-cost_diff / temp):
            current_order = new_order
            current_cost = new_cost
            
            # Update best
            if current_cost < best_cost:
                best_order = current_order.copy()
                best_cost = current_cost
        
        # Cool down
        temp *= cooling_rate
    
    # Calculate transitions
    transitions = []
    for i in range(len(best_order) - 1):
        curr_idx = best_order[i]
        next_idx = best_order[i + 1]
        dist = mood_distance(track_moods[curr_idx], track_moods[next_idx])
        
        transitions.append({
            "from_index": int(curr_idx),
            "to_index": int(next_idx),
            "smoothness": float(max(0, 1 - dist / 2)),
            "distance": float(dist)
        })
    
    # Calculate score
    max_possible_cost = n * 2.0
    flow_score = max(0, 1 - (best_cost / max_possible_cost))
    
    return {
        "optimizedOrder": best_order,
        "flowScore": float(flow_score),
        "transitions": transitions,
        "totalCost": float(best_cost)
    }


@router.get("/algorithms")
async def get_available_algorithms():
    """
    Returns available optimization algorithms and their characteristics.
    """
    return {
        "algorithms": [
            {
                "name": "dynamic_programming",
                "description": "Finds optimal solution using DP. Best quality but slower for large playlists.",
                "time_complexity": "O(n²)",
                "recommended_for": "playlists < 100 tracks"
            },
            {
                "name": "greedy",
                "description": "Fast algorithm that picks nearest next track. Good for quick results.",
                "time_complexity": "O(n²)",
                "recommended_for": "playlists of any size, when speed is priority"
            },
            {
                "name": "simulated_annealing",
                "description": "Probabilistic algorithm that can escape local minima. Good balance of quality and speed.",
                "time_complexity": "O(iterations × n)",
                "recommended_for": "playlists > 100 tracks"
            }
        ]
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "playlist_optimization"
    }