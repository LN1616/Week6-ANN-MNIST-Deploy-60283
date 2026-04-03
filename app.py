import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a prediction.")

model_path = '60283_mnist_model.keras'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure the model is saved correctly.")
else:
    model = tf.keras.models.load_model(model_path)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Convert to grayscale
        img = img.convert('L')

        # Resize to 28x28 pixels
        img = img.resize((28, 28))

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize pixel values to [0, 255] to [0, 1]
        img_array = img_array.astype("float32") / 255.0

        # Reshape for model prediction (add batch dimension)
        img_array = img_array.reshape(1, 28, 28)

        # Make a prediction
        prediction = model.predict(img_array)

        # Get the predicted digit
        predicted_digit = np.argmax(prediction)

        st.success(f"The model predicts the digit is: **{predicted_digit}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}. Please ensure the uploaded image is valid and the model is correctly loaded.")
