import os
import cv2
import joblib
import numpy as np
import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_option_menu import option_menu
# import sklearn
# sklearn.metrics._scorer._passthrough_scorer = None

# Load the phishing detection model using a relative path
model_1 = os.path.join(os.path.dirname(__file__), 'phish_url_model.joblib')
model1 = joblib.load(model_1)

model_2 = os.path.join(os.path.dirname(__file__), 'TfIdf_Vectorizer.joblib')
model2 = joblib.load(model_2)

model_3 = os.path.join(os.path.dirname(__file__), 'Logistics_Regression_Model.joblib')
model3 = joblib.load(model_3)

with st.sidebar:
    selected = option_menu(
        "NAVIGATIONS",
        ['E-Text Phishing Detection', 'QR Phishing Detection'],
        menu_icon='browser-safari',
        icons=['envelope-fill', 'qr-code'],
        default_index=0,
        styles={
            "nav-link": {"font-weight": "bold"},
            "nav-link-selected": {"font-weight": "bold"}
        }
    )
    
st.markdown("<h1 style='text-align: center; color: #00FFFF;'>🔍 Phish Up 🔵</h1>", unsafe_allow_html=True)

# Function to predict email text
def predict_email_text(text):
    text_vectorized = model2.transform([text])
    prediction = model3.predict(text_vectorized)
    return prediction

if selected == 'E-Text Phishing Detection':
    st.title("Email Text Phishing Detection")
    with st.form("email_form"):
        user_input = st.text_area("Enter an email text to check if it's phishing:")
        submitted = st.form_submit_button("Predict")
    
        if submitted and user_input.strip():
            sentiment = predict_email_text(user_input)
    
            if sentiment == 1:
                st.error("!! The given text is a phishing email !! 🎣")
                rain(emoji="🎣", font_size=20, falling_speed=2, animation_length="infinite")
            else:
                st.success("The given text is a safe email!")
                rain(emoji="✅", font_size=20, falling_speed=2, animation_length="infinite")
        elif submitted:
            st.warning("Please enter a text before predicting.")
            
elif selected == 'QR Phishing Detection':
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
    
            prediction = model.predict([data])

             # 0 indicates phishing
            if prediction[0] == 0:
                st.write("Result: **Phishing QR Code**")
            else:
                st.write("Result: **Safe QR Code**")
        else:
            st.write("No QR code detected.")
