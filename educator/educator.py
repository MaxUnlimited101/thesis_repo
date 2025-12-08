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
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import io
from datetime import datetime
from fastapi.staticfiles import StaticFiles


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


async def calculate_overall_statistics():
    # Calculate overall statistics from predictions_log
    stats = {}
    async with predictions_lock:
        for entry in predictions_log:
            student_id, timestamp, predictions = entry
            if student_id not in stats:
                stats[student_id] = {k: 0.0 for k in predictions.keys()}
                stats[student_id]['count'] = 0
            for k, v in predictions.items():
                stats[student_id][k] += v
            stats[student_id]['count'] += 1
        
        # Average the predictions
        for student_id, values in stats.items():
            count = values.pop('count')
            for k in values.keys():
                values[k] /= count
    
    return stats


async def generate_plot(student_id: str = None, cumulative: bool = False):
    """Generate a stacked area chart showing emotion distribution over time"""
    
    async with predictions_lock:
        if not predictions_log:
            return None
        
        # Filter by student_id if provided
        if student_id:
            filtered_data = [(sid, ts, preds) for sid, ts, preds in predictions_log if sid == student_id]
        else:
            filtered_data = predictions_log
        
        if not filtered_data:
            return None
        
        # Sort by timestamp
        filtered_data.sort(key=lambda x: x[1])
        
        # Extract timestamps and predictions
        timestamps = [entry[1] for entry in filtered_data]
        
        # Get all emotion keys from first entry
        emotion_keys = list(filtered_data[0][2].keys())
        
        # Create arrays for each emotion
        emotion_data = {key: [] for key in emotion_keys}
        for _, _, predictions in filtered_data:
            for key in emotion_keys:
                emotion_data[key].append(predictions.get(key, 0.0))
        
        # Convert to numpy arrays for stacking
        emotion_arrays = [np.array(emotion_data[key]) for key in emotion_keys]

        if cumulative:
            emotion_arrays = [np.cumsum(arr) for arr in emotion_arrays]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create time indices (session index)
        time_indices = list(range(1, len(timestamps) + 1))
        
        # Define colors for each emotion (matching the image)
        colors = {
            'neutral': '#5B9BD5',
            'happy': '#FF8C42',
            'sad': '#70AD47',
            'surprise': '#A682C4',
            'fear': '#C09482',
            'disgust': '#999999',
            'angry': '#C4D85F',
            'contempt': '#4CC7C7'
        }
        
        # Create color list in order of emotion_keys
        color_list = [colors.get(key.lower(), '#CCCCCC') for key in emotion_keys]
        
        # Create stacked area chart
        ax.stackplot(time_indices, *emotion_arrays, labels=emotion_keys, colors=color_list, alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Time (session index)', fontsize=12)
        ylabel = 'Cumulative Emotion Count' if cumulative else 'Emotion Probability'
        ax.set_ylabel(ylabel, fontsize=12)

        title_suffix = f" - {student_id}" if student_id else " - All Students"
        title_prefix = "Cumulative " if cumulative else ""
        ax.set_title(f'{title_prefix}Emotion Distribution Over Time{title_suffix}', fontsize=14, pad=20)
        
        if not cumulative:
            ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                 framealpha=0.9, fontsize=10)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return buf



@app.get("/api/plot/{student_id}")
async def get_student_plot(student_id: str, type: str = "regular"):
    """Get emotion distribution plot for a specific student"""
    cumulative = (type == "cumulative")
    plot_buffer = await generate_plot(student_id, cumulative=cumulative)
    
    if plot_buffer is None:
        return JSONResponse({"error": "No data available"}, status_code=404)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(plot_buffer, media_type="image/png")

@app.get("/api/plot")
async def get_all_students_plot(type: str = "regular"):
    """Get emotion distribution plot for all students combined"""
    cumulative = (type == "cumulative")
    plot_buffer = await generate_plot(cumulative=cumulative)
    
    if plot_buffer is None:
        return JSONResponse({"error": "No data available"}, status_code=404)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(plot_buffer, media_type="image/png")


@app.post("/api/emotions")
async def receive_emotions(request: Request):
    try:
        data = await request.json()
        timestamp = int(time.time())

        async with predictions_lock:
            predictions_log.append((data['id'], timestamp, data['predictions']))
        
        print(f"Received data from {data['id']} at {timestamp}: {data['predictions']}")
            
        return JSONResponse(
            {"status": "ok"},
            status_code=200,
        )
        
    except Exception as e:
        print("Error:", e)
        return JSONResponse({"error": str(e)}, status_code=400)



@app.get("/api/statistics")
async def get_statistics():
    """Get overall statistics for the dashboard"""
    async with predictions_lock:
        student_counts = defaultdict(int)
        for student_id, _, _ in predictions_log:
            student_counts[student_id] += 1
        
        return {
            "total_students": len(student_counts),
            "total_predictions": len(predictions_log),
            "active_sessions": len(student_counts),
            "students": dict(student_counts)
        }


@app.get("/health")
async def health_check():
    return {"message": "ok"}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    init()
    uvicorn.run("educator:app", host="0.0.0.0", port=8001, reload=True)
