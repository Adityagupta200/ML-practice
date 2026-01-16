import json
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from textblob import TextBlob
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import accuracy_score
import tensorflow as tf

def preprocess_text(sentence):
    """
    This function preprocesses text data for bias detection.
    Args:
        sentence: A string containing the text to be preprocessed.
    Returns:
        A list of preprocessed tokens.
    """
    # Lowercase all text
    sentence = sentence.lower()

    # Remove punctuation and special characters
    punctuations = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
    for char in punctuations:
        sentence = sentence.replace(char, "")

    # Remove stop words (replace with your stop word list)
    stop_words = ["a", "an", "the", "is", "of", "to"]
    sentence = " ".join([word for word in sentence.split() if word not in stop_words])

    # Optional: Stemming or Lemmatization (implement if needed)
    sentiment = TextBlob(sentence).sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

    return sentence, sentiment or 0

def predict_bias(sentence,context):
    sentence_tokens, sentence_sentiment_score = preprocess_text(sentence)
    context_tokens, context_sentiment_score = preprocess_text(context)

    # Load the saved vectorizer
    sentence_vectorizer_tfidf = joblib.load('tfidf_vectorizer.pkl')
    sentence_tfidf_features = sentence_vectorizer_tfidf.transform([sentence_tokens]).toarray()

    context_vectorizer_tfidf = joblib.load('context_tfidf_vectorizer.pkl')
    context_tfidf_features = context_vectorizer_tfidf.transform([context_tokens]).toarray()
    # Load the trained model
    loaded_model = load_model('bias_type_model.keras')

    # Reshape sentence_sentiment_score to make it 2D (as models generally expect 2D input)
    sentence_sentiment_score = np.array([[sentence_sentiment_score]])  # Convert to a 2D array
    context_sentiment_score = np.array([context_sentiment_score])

    # Make predictions using the model
    predictions = loaded_model.predict([sentence_tfidf_features, sentence_sentiment_score, context_tfidf_features, context_sentiment_score])

    # Get the predicted bias type (if it’s a classification model, you can use argmax)
    predicted_bias_type = predictions.argmax(axis=1)  # Get the predicted bias type index
    return predicted_bias_type

# predicted_bias = predict_bias("She does not have to worry about child predators.", "The schoolgirl is walking down the street.")
# le = joblib.load('label_encoder.pkl')  # Load the encoder
# print("Predicted Bias Type:", le.inverse_transform(predicted_bias))
# print(le.classes_)

# Load testing features and labels
sentence_tfidf_features_test = joblib.load('sentence_tfidf_features_test.pkl')
context_tfidf_features_test = joblib.load('context_tfidf_features_test.pkl')
sentence_sentiment_scores_test_filtered = joblib.load('sentence_sentiment_scores_test_filtered.pkl')
context_sentiment_scores_test_filtered = joblib.load('context_sentiment_scores_test_filtered.pkl')
bias_type_test_filtered_encoded = joblib.load('bias_type_test_filtered_encoded.pkl')

# Load the trained model
model = tf.keras.models.load_model('bias_type_model.keras')

# Prepare testing inputs
test_inputs = [
    np.array(sentence_tfidf_features_test),
    np.array(sentence_sentiment_scores_test_filtered).reshape(-1, 1),
    np.array(context_tfidf_features_test),
    np.array(context_sentiment_scores_test_filtered).reshape(-1, 1)
]

# Make predictions on the testing data
test_predictions = model.predict(test_inputs)
predicted_classes = np.argmax(test_predictions, axis=1)

# Calculate and print accuracy
test_accuracy = accuracy_score(bias_type_test_filtered_encoded, predicted_classes)
print(f"Test Accuracy: {test_accuracy:.2f}")
