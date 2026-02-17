
from pydantic import BaseModel, Field, ValidationError
from typing import List

# This is your schema from Phase 4
class PredictionRequest(BaseModel):
    features: List[List[float]] = Field(..., min_items=1)
    batch_id: str = "unknown"

# TEST 1: Valid Data
print("--- Test 1: Valid Data ---")
good_data = {"features": [[0.1, 0.2, 0.3]], "batch_id": "test_001"}
request = PredictionRequest(**good_data)
print(f"Success! Features: {request.features}")

# TEST 2: Invalid Data (The 'Wait, what?' test)
print("\n--- Test 2: Invalid Data ---")
try:
    bad_data = {"features": [1, 2], "batch_id": 123} # Wrong types
    PredictionRequest(**bad_data)
except ValidationError as e:
    print(f"Pydantic blocked it! Errors found:\n{e.json()}")

# LAB TASK: Change 'features' in bad_data to [[0.1, "high_risk"]]. 
# Does Pydantic catch that the inner list has a string?
