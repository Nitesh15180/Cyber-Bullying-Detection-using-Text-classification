import streamlit as st
import pickle
from textblob import TextBlob
import pytesseract
from PIL import Image

# Load the trained model and vectorizer
try:
    with open(r"C:\Users\DELL\PycharmProjects\pro_2\pythonProject1\stopwords.txt", "r") as file:
        stopwords = file.read().splitlines()

    # Load the vectorizer and model
    with open(r"C:\Users\DELL\PycharmProjects\pro_2\pythonProject1\tfidvectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    with open(r"C:\Users\DELL\PycharmProjects\pro_2\pythonProject1\LinearSVC.sav", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
except Exception as e:
    st.error(f"An error occurred while loading files: {str(e)}")

# Initialize session state for prediction counts
if 'bullying_count' not in st.session_state:
    st.session_state.bullying_count = 0
if 'non_bullying_count' not in st.session_state:
    st.session_state.non_bullying_count = 0

# Function to predict bullying
def predict_bullying(text):
    try:
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)
        return 'Bullying' if prediction[0] == 1 else 'Non Bullying'
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

# Function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function for text extraction from image
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"An error occurred while extracting text: {str(e)}")
        return ""

# Streamlit interface
st.title("Cyberbullying Detection")

# Apply custom CSS for font size and text box colors
st.markdown("""
    <style>
    .big-text-area textarea {
        font-size: 22px;
        height: 200px;
        width: 100%;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        margin-top: 20px;
        width: 100%;
    }
    .red-box {
        color: white;
        background-color: red;
        border: 2px solid darkred;
    }
    .green-box {
        color: white;
        background-color: green;
        border: 2px solid darkgreen;
    }
    .neutral-box {
        color: black;
        background-color: lightgrey;
        border: 2px solid grey;
    }
    </style>
    """, unsafe_allow_html=True)

# Render the text area with a custom HTML container
st.markdown('<div class="big-text-area">', unsafe_allow_html=True)
user_input = st.text_area("Enter the text you want to analyze:", placeholder="Type here...", max_chars=1000, key="text_area", help="Enter the text you want to analyze")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Predict"):
    if user_input:
        result = predict_bullying(user_input)
        sentiment = analyze_sentiment(user_input)
        if result:
            if result == 'Bullying':
                st.session_state.bullying_count += 1
                box_class = 'red-box'
            else:
                st.session_state.non_bullying_count += 1
                box_class = 'green-box'
            st.markdown(f'<div class="prediction-box {box_class}">**Prediction:** {result}</div>', unsafe_allow_html=True)
            st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")

# Display prediction statistics
st.sidebar.header("Prediction Statistics")
st.sidebar.write(f"Total Bullying Predictions: {st.session_state.bullying_count}")
st.sidebar.write(f"Total Non-Bullying Predictions: {st.session_state.non_bullying_count}")

# Collect user feedback
st.sidebar.header("Feedback")
feedback = st.sidebar.text_area("Share your feedback on the prediction:", placeholder="Your feedback...")
if st.sidebar.button("Submit Feedback"):
    if feedback:
        st.sidebar.write("Thank you for your feedback!")
    else:
        st.sidebar.write("Please enter your feedback before submitting.")

# Add functionality for image-based text extraction
st.sidebar.header("Image Text Extraction")
st.sidebar.write("Upload an image to extract and analyze text.")

# Allow users to upload an image file for text extraction
uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Read the image
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image.", use_column_width=True)
    text = extract_text_from_image(image)
    st.sidebar.write("Extracted Text:")
    st.sidebar.write(text)
    # Predict based on the extracted text
    if text:
        result = predict_bullying(text)
        sentiment = analyze_sentiment(text)
        if result:
            if result == 'Bullying':
                box_class = 'red-box'
            else:
                box_class = 'green-box'
            st.sidebar.markdown(f'<div class="prediction-box {box_class}">**Prediction from Image Text:** {result}</div>', unsafe_allow_html=True)
            st.sidebar.write(f"Sentiment: {sentiment}")
