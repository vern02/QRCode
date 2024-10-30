import cv2
import numpy as np
import streamlit as st
import joblib
import os

# Load the phishing detection model using a relative path
model_path = os.path.join(os.path.dirname(__file__), 'phish_url_model.joblib')
model = joblib.load(model_path)

st.title("QR Code Scanner with Phishing Detection")

# Capture QR code from the camera
image = st.camera_input("Show QR code")

if image is not None:
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Create a QR code detector instance
    detector = cv2.QRCodeDetector()

    # Detect and decode the QR code
    data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

    if data:
        st.write("Scanned Data: ", data)

        # Prepare data for model prediction
        prediction = model.predict([data])  # Adjust this line if your model requires a specific input format
        
        # Display the result
        if prediction[0] == 0:  # 0 indicates phishing
            st.write("Result: **Phishing QR Code**")
        else:
            st.write("Result: **Safe QR Code**")
    else:
        st.write("No QR code detected.")
