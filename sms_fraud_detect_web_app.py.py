# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:59:37 2024

@author: ABC
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
#loading saved model

import streamlit as st

# Paths to the model and vectorizer
model_path = 'C:/Users/ABC/Downloads/SMS FRAUD WEB/sms.sav'
vectorizer_path = 'C:/Users/ABC/Downloads/SMS FRAUD WEB/vectorizer.pkl'

# Load the saved model and vectorizer
loaded_model = pickle.load(open(model_path, 'rb'))
feature_extraction = pickle.load(open(vectorizer_path, 'rb'))

# Function to predict spam
def predict_spam(message, threshold=0.5):
    message_features = feature_extraction.transform([message])
    probabilities = loaded_model.predict_proba(message_features)[0]
    if probabilities[0] >= threshold:
        return 0  # Spam (because 0 is spam in my dataset)
    else:
        return 1  # Not spam (because 1 is non-spam in my dataset)

# Define the Streamlit app
def main():
    # Giving title
    st.title('SMS Spam Detection')
    
    st.write("""
    ### Enter the SMS message:
    """)
    
    # Input for the model
    message = st.text_area('Message')
    threshold = st.slider('Spam Threshold', 0.0, 1.0, 0.5)

    if st.button('Predict'):
        result = predict_spam(message, threshold)
        if result == 0:
            st.write('Prediction: This is Spam SMS')
        else:
            st.write('Prediction: This SMS is not Spam')

if __name__ == '__main__':
    main()
