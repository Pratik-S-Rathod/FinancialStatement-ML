import streamlit as st
import pickle

model_path = 'final.pkl'
with open(model_path, 'rb') as file:
    final = pickle.load(file)

cv_path = 'cv.pkl'
with open(cv_path, 'rb') as file:
    cv= pickle.load(file)

# Function to preprocess input value
def preprocess_val(val):
    return val

# Streamlit UI
st.title('Sentiment Analysis with Logistic Regression')

# Input text box for the user to enter a review
user_input = st.text_area('Enter a review:', '')

# Preprocess the input text
preprocessed_input = preprocess_val(user_input)

# Make predictions on user input
if st.button('Predict'):
    if not user_input or not cv.transform([preprocessed_input]).nnz:
        st.warning('Please enter ')
    else:
        try:
            # Vectorize the preprocessed text using the loaded vectorizer
            user_input= cv.transform([preprocessed_input])

            # Make a prediction using the loaded model
            prediction = final.predict(user_input)

            # Display the prediction label
            sentiment_labels = ['negative', 'neutral', 'positive']

            if prediction[0] in [0, 1, 2]:  # Check if the prediction is within the expected range
                st.write('Prediction:', sentiment_labels[prediction[0]])

        except Exception as e:
            st.error(f"An error occurred: {e}")
