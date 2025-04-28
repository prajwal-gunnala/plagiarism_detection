import streamlit as st
import pandas as pd
import string
import pickle
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Plagiarism Checker",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
        text-align: center;
    }
    .subtitle {
        font-size: 20px;
        margin-bottom: 30px;
        color: #424242;
        text-align: center;
    }
    .result-box {
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .plagiarized {
        background-color: #ffcdd2;
        color: #c62828;
    }
    .not-plagiarized {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        color: #616161;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📝 Plagiarism Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Check your text for plagiarism using machine learning</div>', unsafe_allow_html=True)

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove stop words (without using nltk to avoid import issues)
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                 "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
                 "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                 "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                 "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
                 "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                 "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
                 "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
                 "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
                 "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
                 "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                 "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Load and prepare the dataset
@st.cache_data
def load_dataset():
    # Load dataset from GitHub
    url = "https://raw.githubusercontent.com/prajwal-gunnala/plagiarism_detection/main/dataset.csv"
    try:
        data = pd.read_csv(url)
        return data
    except:
        st.error("Failed to load dataset from GitHub. Using a sample dataset instead.")
        # Create a sample dataset as fallback
        data = {
            "source_text": [
                "Researchers have discovered a new species of butterfly in the Amazon rainforest.",
                "The moon orbits the Earth in approximately 27.3 days.",
                "Water is composed of two hydrogen atoms and one oxygen atom.",
                "The history of Rome dates back to 753 BC.",
                "Pluto was once considered the ninth planet in our solar system."
            ],
            "plagiarized_text": [
                "Scientists have found a previously unknown butterfly species in the Amazon jungle.",
                "Our natural satellite takes around 27.3 days to complete one orbit around Earth.",
                "H2O consists of 2 hydrogen atoms and 1 oxygen atom.",
                "Rome has a long history that can be traced back to 753 BC.",
                "In the past, Pluto was classified as the ninth planet of our solar system."
            ],
            "label": [1, 1, 1, 1, 1]
        }
        # Add non-plagiarized examples
        for i in range(5):
            data["source_text"].append(f"Original text example {i+1} that is completely unique.")
            data["plagiarized_text"].append(f"Another unique text sample {i+1} with different content.")
            data["label"].append(0)

        return pd.DataFrame(data)

# Train model
@st.cache_resource
def train_model(data):
    # Preprocess the data
    data["source_text"] = data["source_text"].apply(preprocess_text)
    data["plagiarized_text"] = data["plagiarized_text"].apply(preprocess_text)

    # Create features
    tfidf_vectorizer = TfidfVectorizer()
    combined_texts = data["source_text"] + " " + data["plagiarized_text"]
    X = tfidf_vectorizer.fit_transform(combined_texts)
    y = data["label"]

    # Train model
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)

    return model, tfidf_vectorizer

# Function to detect plagiarism
def detect_plagiarism(input_text, model, vectorizer):
    # Preprocess the input text
    processed_text = preprocess_text(input_text)

    # Vectorize the input text
    vectorized_text = vectorizer.transform([processed_text])

    # Get prediction and probability
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]

    # Calculate confidence percentage
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    confidence_pct = round(confidence * 100, 2)

    return prediction, confidence_pct

# Main app functionality
def main():
    # Load dataset and train model
    with st.spinner("Loading model..."):
        data = load_dataset()
        model, vectorizer = train_model(data)

    # Create layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Text input
        user_text = st.text_area(
            "Enter the text you want to check for plagiarism:",
            height=300,
            placeholder="Paste your text here..."
        )

        # Check button
        if st.button("Check for Plagiarism", type="primary"):
            if not user_text:
                st.warning("Please enter some text to check.")
            else:
                with st.spinner('Analyzing text...'):
                    prediction, confidence = detect_plagiarism(user_text, model, vectorizer)

                # Display result
                if prediction == 1:
                    st.markdown(
                        f'<div class="result-box plagiarized">⚠️ Plagiarism Detected (Confidence: {confidence}%)</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-box not-plagiarized">✅ No Plagiarism Detected (Confidence: {confidence}%)</div>',
                        unsafe_allow_html=True
                    )

    with col2:
        st.markdown("### How it works")
        st.markdown("""
        This plagiarism checker uses machine learning to detect similarities between texts:

        1. **Text Preprocessing**: Removes punctuation, converts to lowercase, and removes stop words
        2. **Vectorization**: Converts text to numerical vectors using TF-IDF
        3. **Classification**: Uses a Support Vector Machine (SVM) model to classify text as plagiarized or original

        The model was trained on a dataset of paired texts (original and plagiarized versions).
        """)

        st.markdown("### Examples to try")
        st.info("**May detect as plagiarized**:\nResearchers have discovered a new species of butterfly in the Amazon rainforest.")
        st.success("**Should detect as original**:\nPracticing yoga regularly can significantly improve flexibility and reduce stress levels.")

    # Footer
    st.markdown('<div class="footer">© 2025 Plagiarism Checker | Created with Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
