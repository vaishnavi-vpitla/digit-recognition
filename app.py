
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

st.title("Handwritten Digit Recognizer")
st.write("Draw a single digit (0-9) inside the box below.")

@st.cache_resource
def load_keras_model():
    # Loads the .h5 model file directly from your workspace folder
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_keras_model()

canvas_result = st_canvas(
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas",
)

if canvas_result is not None and canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    if np.sum(img) > 0:
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        # Reshape for Keras: (1, 28, 28, 1)
        tensor_img = np.expand_dims(img, axis=(0, -1))

        prediction = model.predict(tensor_img)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        st.header(f"Predicted Digit: {predicted_digit}")
        st.subheader(f"Confidence: {confidence:.2f}%")