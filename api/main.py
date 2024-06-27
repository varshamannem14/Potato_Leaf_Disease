from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = {
    "http://localhost",
    "http://localhost:3000",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers = ["*"],
)


MODEL = TFSMLayer("C:/Users/varsha/OneDrive/Desktop/Potato_Disease/models/1", call_endpoint='serving_default')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = read_file_as_image(await file.read())
        print("Image shape:", image.shape)
        img_batch = np.expand_dims(image, 0)
        print("Batch shape:", img_batch.shape)

        # Get predictions
        predictions = MODEL(img_batch)
        print("Predictions type:", type(predictions))
        
        # Debug: Print the structure if predictions is a dict
        if isinstance(predictions, dict):
            print("Prediction keys:", predictions.keys())
            if 'dense_3' in predictions:
                predictions = predictions['dense_3']
            else:
                raise KeyError("Key 'dense_3' not found in predictions dictionary")

        # Ensure predictions is a numpy array
        predictions = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)

        # Return the prediction result
        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)
