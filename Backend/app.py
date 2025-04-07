from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io

app = FastAPI()

# Define image dimensions and class categories
img_width, img_height = 100, 100
data_category = ['aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut', 'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya', 'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach', 'sweet potatoes', 'tobacco', 'waterapple', 'watermelon']

# Load model
with open("../Model/SNNmodel.pkl", "rb") as file:
    model = pickle.load(file)

def preprocess_image(image_bytes):
    """Preprocess the image for prediction."""
    image = tf.keras.utils.load_img(io.BytesIO(image_bytes), target_size=(img_width, img_height))
    image_arr = tf.keras.utils.img_to_array(image)
    image_bat = tf.expand_dims(image_arr, 0)  # shape: (1, 100, 100, 3)
    return image_bat

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        # Preprocess and predict
        input_tensor = preprocess_image(contents)

        predictions = model.predict(input_tensor)
        print("Predictions:", predictions)
        print("Predictions shape:", predictions.shape)

        if predictions.shape[0] == 0:
            return JSONResponse(content={"error": "Empty prediction output from model."}, status_code=500)

        score = tf.nn.softmax(predictions[0])

        if len(data_category) != score.shape[0]:
            return JSONResponse(
                content={"error": f"Mismatch in number of classes. Model returned {score.shape[0]} scores, but data_category has {len(data_category)} items."},
                status_code=500
            )

        predicted_class = data_category[np.argmax(score)]
        confidence = 100 * np.max(score)

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
