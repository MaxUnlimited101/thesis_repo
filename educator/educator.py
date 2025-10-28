# receiver_app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Simulated student database (map IPs to student names)
STUDENTS = {
    "127.0.0.1": "Alex",
    # later you can add other IPs like:
    # "192.168.0.15": "John",
}

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
    uvicorn.run("Educator:app", host="0.0.0.0", port=8001, reload=True)
