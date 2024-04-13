import streamlit as st
import joblib
import numpy as np

# Load the saved models
logreg_model = joblib.load('..Models/LogisticRegression_model.pkl')
# Load other models similarly


def predict(text):
    predictions = {}
    # Make predictions using each model
    predictions['Logistic Regression'] = logreg_model.predict_proba([text])[0]
    # Make predictions using other models similarly
    return predictions


def main():
    # Set the title of the app
    st.title('Depression Prediction')

    # Add a text input for user to enter text
    text = st.text_input('Enter text:', '')

    # Add a button to trigger predictions
    if st.button('Predict'):
        if text:
            # Make predictions
            predictions = predict(text)
            # Display predictions
            st.write('Predictions:')
            for model_name, prediction in predictions.items():
                st.write(f'{model_name}: {prediction[1]:.2f}')
        else:
            st.write('Please enter some text.')

if __name__ == '__main__':
    main()