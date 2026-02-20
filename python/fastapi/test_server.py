from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 1. The Guard (Dependency)
async def simple_security(secret_token: str = Header(...)):
    if secret_token != "phd_to_mlops":
        raise HTTPException(status_code=403, detail="Wrong Token!")
    return secret_token

# 2. The Schema
class SimpleResponse(BaseModel):
    message: str
    processed_val: float

# 3. The Endpoints (The Address Labels)
@app.get("/check")
async def check():
    return {"status": "I am alive"}

@app.post("/compute", response_model=SimpleResponse)
async def compute(val: float, token: str = Header(...)):
    # Here we manually check logic
    if token == "phd_to_mlops":
        return {"message": "Success", "processed_val": val * 2}
    raise HTTPException(status_code=401)

if __name__ == "__main__":
    # This starts the server locally on port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
