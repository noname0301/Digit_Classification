from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image, ImageOps
import io
import os
import socket
import sys
from models.model import LeNet5
import uvicorn
import logging
import pandas as pd

# Add parent directory to Python path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load confusion matrix
confusion_matrix = pd.read_csv("models/confusion_matrix.csv", header=None)
confusion_matrix = confusion_matrix.to_numpy()

# Calculate metrics
accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
f1_score = 2 * precision * recall / (precision + recall)
average_precision = np.mean(precision)
average_recall = np.mean(recall)
average_f1_score = np.mean(f1_score)

app = FastAPI(title="Digit Recognition API", description="API for digit recognition")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_available_port(start_port=8000, max_port=8999):
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

# Get the current directory and model path
model_path = "models/mnist_model.pth"

# Load the PyTorch model
logger.info("Loading model...")
logger.info(f"Looking for model at: {model_path}")


try:
    model = LeNet5()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    model = None
    logger.error(f"Error loading model: {str(e)}")



@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((28, 28))
        # Pad white pixel around image to 32x32
        image = ImageOps.expand(image, border=2, fill='white')


        # Convert to numpy array and normalize
        mnist_mean = 0.1307
        mnist_std = 0.3081

        image_array = np.array(image)
        image_array = 255 - image_array  # Invert colors

        

        image_array = image_array.astype(np.float32) / 255.0
        image_array = (image_array - mnist_mean) / mnist_std

        # plt.imshow(image_array, cmap="gray")
        # plt.show()
        
        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
            confidence = probabilities[0][prediction].item()
        
        logger.info(f"Prediction: {prediction.item()}, Confidence: {confidence:.4f}")
        return {
            "prediction": prediction.item(),
            "confidence": confidence,
            "probabilities": probabilities[0].tolist()
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/confusion_matrix")
async def get_confusion_matrix():
    return {"confusion_matrix": confusion_matrix.tolist(),
            "metrics": {
                "accuracy": accuracy,
                "precision": average_precision,
                "recall": average_recall,
                "f1Score": average_f1_score
            }}


if __name__ == "__main__":
    port = find_available_port()
    logger.info(f"API is available at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 