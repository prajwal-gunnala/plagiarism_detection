import streamlit as st
import nltk
import string
import pickle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
from PIL import Image
import requests
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Plagiarism Checker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

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
    .stTextArea label {
        font-size: 18px;
        font-weight: bold;
        color: #1E88E5;
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
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
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
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Load model and vectorizer
@st.cache_resource
def load_model():
    # For demo purposes, we'll train a simple model on a small dataset
    # In a real application, you'd load your pre-trained model
    
    # Sample dataset
    data = {
        "source_text": [
            "Researchers have discovered a new species of butterfly in the Amazon rainforest.",
            "The moon orbits the Earth in approximately 27.3 days.",
            "Water is composed of two hydrogen atoms and one oxygen atom.",
            "The history of Rome dates back to 753 BC.",
            "Pluto was once considered the ninth planet in our solar system.",
            "Playing musical instruments enhances creativity.",
            "Reading books improves vocabulary and critical thinking skills.",
            "Exercise is essential for maintaining good health.",
            "The Great Wall of China is visible from space.",
            "The Mona Lisa was painted by Leonardo da Vinci."
        ],
        "plagiarized_text": [
            "Scientists have found a previously unknown butterfly species in the Amazon jungle.",
            "Our natural satellite takes around 27.3 days to complete one orbit around Earth.",
            "H2O consists of 2 hydrogen atoms and 1 oxygen atom.",
            "Rome has a long history that can be traced back to 753 BC.",
            "In the past, Pluto was classified as the ninth planet of our solar system.",
            "Musical instrument practice fosters enhanced creativity.",
            "Expanding vocabulary and critical thinking abilities is a benefit of reading books.",
            "Good health maintenance requires regular physical activity.",
            "From space, one can observe the Great Wall of China.",
            "Leonardo da Vinci is the artist who created the Mona Lisa."
        ],
        "label": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    # Add non-plagiarized examples
    for i in range(10):
        data["source_text"].append(f"Original text example {i+1} that is completely unique.")
        data["plagiarized_text"].append(f"Another unique text sample {i+1} with different content.")
        data["label"].append(0)
    
    df = pd.DataFrame(data)
    
    # Preprocess text
    df["source_text"] = df["source_text"].apply(preprocess_text)
    df["plagiarized_text"] = df["plagiarized_text"].apply(preprocess_text)
    
    # Vectorize text
    tfidf_vectorizer = TfidfVectorizer()
    combined_texts = df["source_text"] + " " + df["plagiarized_text"]
    tfidf_vectorizer.fit(combined_texts)
    
    # Train SVM model
    from sklearn.svm import SVC
    X = tfidf_vectorizer.transform(combined_texts)
    y = df["label"]
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)
    
    return model, tfidf_vectorizer

# Load model
with st.spinner('Loading model...'):
    model, tfidf_vectorizer = load_model()

# Function to detect plagiarism
def detect_plagiarism(input_text):
    # Preprocess the input text
    processed_text = preprocess_text(input_text)
    
    # Vectorize the input text
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    
    # Get prediction and probability
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    # Calculate confidence percentage (probability of the predicted class)
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    confidence_pct = round(confidence * 100, 2)
    
    return prediction, confidence_pct

# Main app
col1, col2 = st.columns([2, 1])

with col1:
    # Text input
    user_text = st.text_area(
        "Enter the text you want to check for plagiarism:",
        height=300,
        placeholder="Paste your text here..."
    )
    
    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        check_button = st.button("Check for Plagiarism", use_container_width=True)
        
    with col_btn2:
        clear_button = st.button("Clear Text", use_container_width=True)
        if clear_button:
            user_text = ""
            st.experimental_rerun()

with col2:
    st.markdown("### How it works")
    st.markdown("""
    This plagiarism checker uses machine learning to detect similarities between texts:
    
    1. **Text Preprocessing**: Removes punctuation, converts to lowercase, and removes stop words
    2. **Vectorization**: Converts text to numerical vectors using TF-IDF
    3. **Classification**: Uses a Support Vector Machine (SVM) model to classify text as plagiarized or original
    4. **Confidence Score**: Provides a confidence percentage for the prediction
    
    **Note**: This is a demonstration using a small dataset. Real plagiarism detection systems use much larger databases of content for comparison.
    """)
    
    st.markdown("### Example texts to try")
    st.info("**Plagiarized**: Researchers have discovered a new species of butterfly in the Amazon rainforest.")
    st.success("**Original**: The benefits of daily meditation include reduced stress and improved focus.")

# Display result when check button is clicked
if check_button and user_text:
    with st.spinner('Analyzing text...'):
        prediction, confidence = detect_plagiarism(user_text)
    
    st.markdown("### Result")
    
    if prediction == 1:
        st.markdown(
            f'<div class="result-box plagiarized">⚠️ Plagiarism Detected (Confidence: {confidence}%)</div>',
            unsafe_allow_html=True
        )
        st.warning("This text appears to be plagiarized or contains significant similarities to existing content.")
    else:
        st.markdown(
            f'<div class="result-box not-plagiarized">✅ No Plagiarism Detected (Confidence: {confidence}%)</div>',
            unsafe_allow_html=True
        )
        st.success("This text appears to be original content.")
    
    # Show additional details
    with st.expander("See detailed analysis"):
        st.markdown(f"**Confidence Score**: {confidence}%")
        st.markdown("**Text Length**: " + str(len(user_text)) + " characters")
        st.markdown("**Word Count**: " + str(len(user_text.split())))
        
        # Word cloud placeholder (in a real app, you'd generate an actual word cloud)
        st.markdown("**Top terms in your text:**")
        processed = preprocess_text(user_text)
        word_freq = {}
        for word in processed.split():
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Sort by frequency and get top 10
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        if sorted_words:
            for word, freq in sorted_words:
                st.write(f"- {word}: {freq}")
        else:
            st.write("Text too short for analysis")

# Sidebar with additional info
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This plagiarism checker helps you identify whether your text contains content that may be considered plagiarized.
    
    **Features**:
    - Fast and efficient checking
    - Machine learning-powered analysis
    - Confidence score for results
    
    **Limitations**:
    - Demo version uses a limited dataset
    - Works best for English text
    - Not a substitute for professional plagiarism checking services
    """)
    
    st.markdown("## Tips to avoid plagiarism")
    st.markdown("""
    1. Always cite your sources
    2. Use quotation marks for direct quotes
    3. Paraphrase properly and still cite sources
    4. Use multiple sources for research
    5. Keep detailed notes during research
    """)

# Footer
st.markdown('<div class="footer">© 2025 Plagiarism Checker | Created with Streamlit</div>', unsafe_allow_html=True)
