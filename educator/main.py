import io
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from asyncio import Lock

app = FastAPI()

EMOTIONS = ('angry', 'disgust', 'contempt', 'fear', 'happy', 'neutral', 
            'sad', 'surprise')
TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

predictions = []
lock = Lock()

def init():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")
    
    model = torch.load("model_efb0.pth", weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()
    return model, device
    
    
model, device = init()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict_single(file: UploadFile = File(...), full_name: str = Form(...)) -> None:
    contents = await file.read()
    image = PIL.Image.open(io.BytesIO(contents)).convert('RGB')
    
    with torch.no_grad():
        image_tensor = TRANSFORM(image).unsqueeze(0).to(device)
        
        outputs = model(image_tensor)
        probs = nn.Softmax(dim=1)(outputs)
        _, preds = torch.max(outputs, 1)
        async with lock:
            predictions.append({"prediction": EMOTIONS[preds[0]], "full_name": full_name})
        print(f"Predicted: {EMOTIONS[preds[0]]} for {full_name}")
        return None
            

# @app.post("/predict_batch")
# async def predict_batch(files: List[UploadFile] = File(...)):
#     predictions = []
    
#     for file in files:
#         contents = await file.read()
#         image = PIL.Image.open(io.BytesIO(contents)).convert('RGB')
        
#         with torch.no_grad():
#             image_tensor = TRANSFORM(image).unsqueeze(0).to(device)
            
#             outputs = model(image_tensor)
#             _, preds = torch.max(outputs, 1)
#             predictions.append({"filename": file.filename, "prediction": EMOTIONS[preds[0]]})
    
#     return {"predictions": predictions}