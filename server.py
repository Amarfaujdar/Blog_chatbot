from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from blog_backend import generate_blog
import os

app = FastAPI()

# --- 1. Define Request Schema ---
class TopicRequest(BaseModel):
    topic: str
    audience: str
    tone: str

# --- 2. The API Endpoint ---
@app.post("/generate")
async def generate_endpoint(req: TopicRequest):
    try:
        # Call your existing backend function
        result = generate_blog(req.topic)
        return result
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 3. SERVE STATIC FILES ---
# This is the critical fix for your folder structure.
# It tells the server: "Look inside the 'static' folder for HTML, CSS, and JS"
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Run on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)