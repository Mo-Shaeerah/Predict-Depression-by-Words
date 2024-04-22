# Import Important Libraries

import os
import joblib
import pyttsx3
import numpy as np
from gtts import gTTS
import streamlit as st
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text into audio
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')
    with open('output.mp3', 'rb') as f:
        audio = f.read()
    #os.remove('output.mp3') 
    return audio

# Define Prediction Function
def predict(text):   
    """Predict depression based on input text."""
    # Perform predictions
    predictions = {}
    
    # Load the saved models
    logreg_model = joblib.load('..Models/LogisticRegression_model.pkl')
    nb_model = joblib.load('..Models/multinomial_nb_model.pkl')
    lgbm_model = joblib.load('..Models/LightGBM_model.pkl')
    gradboo_model = joblib.load('..Models/Gradient_Boosting_model.pkl')

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('..Models/tfidf_vectorizer.pkl') 
    # Vectorize the new text data
    text_vectorized = tfidf_vectorizer.transform([text])
    
    predictions['Light-GBM'] = lgbm_model.predict_proba(text_vectorized)[0]
    predictions['Gradient-Boosting'] = gradboo_model.predict_proba(text_vectorized)[0]
    predictions['Logistic Regression'] = logreg_model.predict_proba(text_vectorized)[0]
    predictions['Multinomial Naive Bayes'] = nb_model.predict_proba(text_vectorized)[0]

    return predictions

# Define content for home page
def main():
    # Set A General title for the app
    st.title('`Depression Prediction App 🚥⛈️🧠`')
    st.image('...Images/Project-Image.png', caption='`Project Image 🧠`', use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('...Images/4.png', caption='`Depression-Words e` Charts 🧠`', use_column_width=True)
    with col2:
        st.image('...Images/5.png', caption='`Depression-Words e` Charts🧠`', use_column_width=True)
    
    st.title('`Words Indicating Depre... ⚖️🧉🎡`')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('...Images/1.png', caption='`Depression-Words 🧠`', use_column_width=True)
    with col2:
        st.image('...Images/2.png', caption='`Depression-Words 🧠`', use_column_width=True)
    with col3:
        st.image('...Images/3.png', caption='`Depression-Words🧠`', use_column_width=True)

    st.title("`Image Refer Depre... And Image Not`")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption= 'Refer depression 🤖🤖🤖', use_column_width=True)
    with col2:
        st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= 'Not depression 🚀🚀🚀', use_column_width=True)

    with col3:
        st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption= 'Refer depression 🤖🤖🤖', use_column_width=True)
    with col4:
        st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= 'Not depression 🚀🚀🚀', use_column_width=True)

    # Prediction Part
    st.title('`Start Predicting Depre... 🕹️🕒🔊`')
    text = st.text_input('Enter text 👇🏻👇🏻👇🏻:', '')

    if st.button('`Predict ,🚀🚀🚀`'):
        if text:
            predictions = predict(text)
            counter = 1
            st.write('`Predictions:`')
            for model_name, prediction in predictions.items():
                probability_of_depression = prediction[1]
                if probability_of_depression >= 0.4:
                    st.write(f'{counter}-{model_name} predicts depression with probability: {probability_of_depression:.2%}')
                    st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption='`Image indicating depression 🕸️🕸️🕸️`', use_column_width=True)
                    counter += 1
                    st.subheader("Tips to Improve Mental Well-being :")
                    text_tips = [
                        "Engage in regular physical exercise  🏃‍♂️",
                        "Stay connected with supportive friends  ",
                        "Maintain a healthy diet and get enough sleep  🍎😴",
                        "Seek professional help from a therapist or counselor  ",
                        "Practice relaxation techniques such as deep breathing exercises  😤🎐"
                        ]
                    for i, tip in enumerate(text_tips):
                        st.write(f"{chr(97 + i)}. {tip}")
                    audio_bytes = text_to_audio("\n".join(text_tips))
                    st.audio(audio_bytes, format='audio/wav')
                    audio_generated = True
                else:
                    st.write(f'{counter}-{model_name} predicts No depression with probability: {1 - probability_of_depression:.2%}')
                    st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= '`Image indicating no depression 🚀🚀🚀`', use_column_width=True)
                    counter += 1
        else:
            st.write('`Please enter some text...... 🪔🪔🪔`')

    st.title('`Disclaimer 📌✂️🌡️`')
    st.write("""
             **Disclaimer:**
             
             1. **`Accuracy of Predictions ⚖️⚖️⚖️:`** 
             
             The predictions provided by this app are based on machine learning models trained on available data. 
            
    
             2. **`No Medical Diagnosis 🩺💉😷:`** 
             
             This app does not provide medical diagnosis or treatment recommendations. 
             
             
             3. **`Technical Issues 🧪🧪🧪:`**  
            
             I cannot guarantee uninterrupted access or absence of technical errors.
             I apologize for any inconvenience caused by such issues.

    """)
    

    # Discover the Developer
    developer_name = "Mohammed``` Shaeerah"
    developer_github = "https://github.com/Mo-Shaeerah"
    
    st.title("")
    st.title("")
    st.markdown(f"Created e` 🧠🧠🧠 By: ↪🏹⁀➴↪ [{developer_name}]({developer_github})")

# A page contain the images of the app    
def page_images():
    st.title('`Depression Images Of This APP🧲🧲`')
    st.image('...Images/Project-Image.png', caption='`Project Image 🧠`', use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image('...Images/4.png', caption='`Depression-Words e` Charts 🧠`', use_column_width=True)
    with col2:
        st.image('...Images/5.png', caption='`Depression-Words e` Charts🧠`', use_column_width=True)
    st.title('`Words Indicating Depre... ⚖️🧉🎡`')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('...Images/1.png', caption='`Depression-Words 🧠`', use_column_width=True)
    with col2:
        st.image('...Images/2.png', caption='`Depression-Words 🧠`', use_column_width=True)
    with col3:
        st.image('...Images/3.png', caption='`Depression-Words🧠`', use_column_width=True)

    st.title("`Image Refer Depre... And Image Not`")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption= 'Refer depression 🤖🤖🤖', use_column_width=True)
    with col3:
        st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= 'Not depression 🚀🚀🚀', use_column_width=True)

    with col2:
        st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption= 'Refer depression 🤖🤖🤖', use_column_width=True)
    with col4:
        st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= 'Not depression 🚀🚀🚀', use_column_width=True)



    # Discover the Developer
    developer_name = "Mohammed``` Shaeerah"
    developer_github = "https://github.com/Mo-Shaeerah"
    
    st.title("")
    st.title("")
    st.markdown(f"Created e` 🧠🧠🧠 By: ↪🏹⁀➴↪ [{developer_name}]({developer_github})")

# Prediction page
def page_predictions():
    st.title('`Start Predicting Depre... 🕹️🕒🔊`')
    st.image('...Images/Project-Image.png', caption='`Project Image 🧠`', use_column_width=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption= 'Refer depression 🤖🤖🤖', use_column_width=True)
    with col2:
        st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= 'Not depression 🚀🚀🚀', use_column_width=True)

    with col3:
        st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption= 'Refer depression 🤖🤖🤖', use_column_width=True)
    with col4:
        st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= 'Not depression 🚀🚀🚀', use_column_width=True)

    text = st.text_input('Enter text 👇🏻👇🏻👇🏻:', '')

    if st.button('`Predict ,🚀🚀🚀`'):
        if text:
            predictions = predict(text)
            counter = 1
            st.write('`Predictions:`')
            for model_name, prediction in predictions.items():
                probability_of_depression = prediction[1]
                if probability_of_depression >= 0.4:
                    st.write(f'{counter}- {model_name} predicts depression with probability: {probability_of_depression:.2%}')
                    st.image('https://ideas.ted.com/wp-content/uploads/sites/3/2021/11/FINAL_DepressionAndTrauma.jpg', caption='`Image indicating depression 🕸️🕸️🕸️`', use_column_width=True)
                    counter += 1
                    st.subheader("Tips to Improve Mental Well-being :")
                    text_tips = [
                        "Engage in regular physical exercise  🏃‍♂️",
                        "Stay connected with supportive friends  ",
                        "Maintain a healthy diet and get enough sleep  🍎😴",
                        "Seek professional help from a therapist or counselor  ",
                        "Practice relaxation techniques such as deep breathing exercises  😤🎐"
                        ]
                    for i, tip in enumerate(text_tips):
                        st.write(f"{chr(97 + i)}. {tip}")
                    audio_bytes = text_to_audio("\n".join(text_tips))
                    st.audio(audio_bytes, format='audio/wav')
                    audio_generated = True
                else:
                    st.write(f'{counter}- {model_name} predicts No depression with probability: {1 - probability_of_depression:.2%}')
                    st.image('https://miro.medium.com/v2/resize:fit:1400/1*pys9J8ja_aG-3_jaeT_Ycg.jpeg', caption= '`Image indicating no depression 🚀🚀🚀`', use_column_width=True)
                    counter += 1
        else:
            st.write('`Please enter some text...... 🪔🪔🪔`')


        
    # Discover the Developer
    developer_name = "Mohammed``` Shaeerah"
    developer_github = "https://github.com/Mo-Shaeerah"
    
    st.title("")
    st.title("")
    st.markdown(f"Created e` 🧠🧠🧠 By: ↪🏹⁀➴↪ [{developer_name}]({developer_github})")

# Disclamir page
def display_disclaimer_page():
    # Display the disclaimer page content
    st.title('`Disclaimer 📌✂️🌡️`')
    st.write("""
    **Disclaimer:**

    1. **`Accuracy of Predictions ⚖️⚖️⚖️:`** 
             
            The predictions provided by this app are based on machine learning models trained on available data. 
            
            While I strive for accuracy, these predictions should not be considered definitive
             or a substitute for professional medical advice.
    
    """)
    # Display the image for point 1
    st.image('https://static1.abbyy.com/abbyycommedia/25267/9062e_smm_accountingtoday_blog.png', use_column_width=True)

    st.write("""
    2. **`No Medical Diagnosis 🩺💉😷:`** 
             
            This app does not provide medical diagnosis or treatment recommendations. 
             
            It is intended for informational purposes only. Users should consult
              a qualified healthcare professional for personalized medical advice.

    """)
    # Display the image for point 2
    st.image('https://www.mh-m.org/wp-content/uploads/2017/06/medical-and-health-2.jpg')

    st.write("""
    3. **`Technical Issues 🧪🧪🧪:`** 
             
            While I make every effort to ensure the smooth functioning of this app, 
            
            I cannot guarantee uninterrupted access or absence of technical errors.
             I apologize for any inconvenience caused by such issues.

    """)
    # Display the image for point 3
    st.image('https://s31606.pcdn.co/wp-content/uploads/2016/05/prevent-meeting-technology-problems.jpg')


    # Discover the Developer
    developer_name = "Mohammed``` Shaeerah"
    developer_github = "https://github.com/Mo-Shaeerah"
    
    st.title("")
    st.title("")
    st.markdown(f"Created e` 🧠🧠🧠 By: ↪🏹⁀➴↪ [{developer_name}]({developer_github})")


# Render sidebar
st.sidebar.title('`Navigation 🔭🗺️🗺️`')
selected_page = st.sidebar.radio("**`Go to 🌊🌊🌊`**", ["Depression-Home", "Depression-Images", "Depression-Predictions", "Disclaimer"])

# Display selected page
if selected_page == "Depression-Home":
    main()

elif selected_page == "Depression-Images":
    page_images()

elif selected_page == "Depression-Predictions":
    page_predictions()

elif selected_page == "Disclaimer":
    display_disclaimer_page()


# Set -----
import time
st.sidebar.header("`About The Developer 🦅🦅🦅`")
if st.sidebar.button('The AI Developer 🧙'):
    st.toast('Data Scientist 🕵️')
    time.sleep(1)
    st.toast('GP Medical Doctor 👨🏻‍⚕️')
    time.sleep(1)
    st.toast('Curious About Science 🔎')
    time.sleep(1)
    st.toast('Writes Philosophy & Poetry 🖋')
    time.sleep(1)
    st.toast('Mohammed``` Shaeerah 😎')
    
# Discover the Developer
developer_name = "Mohammed``` Shaeerah"
developer_github = "https://github.com/Mo-Shaeerah"
st.sidebar.markdown(f"Created by 🏹⁀➴ [{developer_name}]({developer_github})")

#if __name__ == '__main__':
#    main()

# Function to get the last update date
def get_last_update_date():
    return datetime(2024, 4, 22)

# Get the last update date and today's date
last_update_date, today_date = get_last_update_date(), datetime.today()
st.sidebar.header('`App Dates & Times ⏲⏲⏲`')

# Create buttons in the sidebar to display the dates upon clicking
if st.sidebar.button("See Today's Date 🍃"):
    st.sidebar.write(f"`Today's Date 👉🏼👉🏼`   {today_date.strftime('%Y-%m-%d')}")
if st.sidebar.button("See Last App-Update Date 🥏"):
    st.sidebar.write(f"`Last Update Date 👉🏼👉🏼`   {last_update_date.strftime('%Y-%m-%d')}")

# Project Links
st.sidebar.header("`Project Links & LinkedIn 🎡🎡🎡`")
# LinkedIn profile link
linkedin_link = "https://www.linkedin.com/in/mo-sa-shaeerah/"
st.sidebar.markdown(f"[LinkedIn Profile ⌨️]({linkedin_link})")
# GitHub project link
github_link = "https://github.com/Mo-Shaeerah/Predict-Depression-by-Words-"
st.sidebar.markdown(f"[GitHub Project-Link 🔗]({github_link})")
# Kaggle project link
kaggle_link = "https://www.kaggle.com/code/mohammedsalf/predict-depression-f-sentiment"
st.sidebar.markdown(f"[Kaggle Project-Link 🖇️]({kaggle_link})")
# Colab project link
colab_link = "https://colab.research.google.com/drive/1jeqtXqIO-okzJudya5qWJnNwr5BgCODg?usp=sharing"
st.sidebar.markdown(f"[Colab Project-Link 🖥️]({colab_link})")