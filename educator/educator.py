# receiver_app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import pinggy
from contextlib import asynccontextmanager


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


# Simulated student database (map IPs to student names)
STUDENTS = {
    "127.0.0.1": "Alex",
    # later you can add other IPs like:
    # "192.168.0.15": "John",
}


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
        student_name = STUDENTS.get(client_ip, "Unknown student")

        # Print info
        print(f"\n📩 Received emotions from {student_name} ({client_ip})")
        for emotion, value in data.items():
            print(f"   {emotion:<10}: {value:.3f}")
        print("-" * 40)

        return JSONResponse(
            {"status": "ok", "student": student_name, "received": data},
            status_code=200,
        )
    except Exception as e:
        print("Error:", e)
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/")
def root():
    return {"message": "Emotion receiver server running"}

if __name__ == "__main__":
    init()
    uvicorn.run("educator:app", host="0.0.0.0", port=8001, reload=True)
