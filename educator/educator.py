# receiver_app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pinggy
from contextlib import asynccontextmanager
from asyncio import Lock
import time
import os
from fastapi.responses import FileResponse


TOKEN = "" # Will be set in init()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the tunnel when the app starts
    init()
    tunnel = setup_tunnel(TOKEN)
    print("Tunnel is running.")
    yield
    # Cleanup when the app stops
    tunnel.stop()
    print("Tunnel stopped.")


app = FastAPI(lifespan=lifespan)
predictions_log = []
predictions_lock = Lock()


def init():
    # read token.txt
    try:
        with open("token.txt", "r") as f:
            global TOKEN
            TOKEN = f.read().strip()
            print(f"Token loaded: {TOKEN}")
    except FileNotFoundError:
        print("ERROR: token.txt not found. Exiting...")
        exit(1)
        

def setup_tunnel(token):
    # Start an HTTP tunnel forwarding traffic to localhost on port 8001
    tunnel = pinggy.start_tunnel(forwardto="localhost:8001", token=token)
    print(f"Tunnel URL: {tunnel.urls[0]}")
    return tunnel


@app.post("/api/emotions")
async def receive_emotions(request: Request):
    try:
        data = await request.json()
        client_ip = request.client.host
        timestamp = int(time.time())

        async with predictions_lock:
            predictions_log.append((client_ip, timestamp, data))
        
        print(f"Received data from {client_ip} at {timestamp}: {data}")
            
        return JSONResponse(
            {"status": "ok"},
            status_code=200,
        )
        
    except Exception as e:
        print("Error:", e)
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/health")
async def health_check():
    return {"message": "ok"}


@app.get("/")
async def root():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(path):
        return JSONResponse({"error": "index.html not found"}, status_code=404)
    return FileResponse(path, media_type="text/html")


if __name__ == "__main__":
    init()
    uvicorn.run("educator:app", host="0.0.0.0", port=8001, reload=True)
