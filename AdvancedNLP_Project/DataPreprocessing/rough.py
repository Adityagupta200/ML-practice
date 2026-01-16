import json
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import numpy as np
from textblob import TextBlob
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import optuna
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import joblib
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM

with open("C:/Users/abc/Downloads/StereoSet.json", "r") as f:
    data = json.load(f)

def preprocess_text(sentences):  # Function definition with correct indentation
    """
    This function preprocesses text data for bias detection.

    Args:
        sentences: A string containing the text to be preprocessed.

    Returns:
        A list of preprocessed tokens.
    """

    # Lowercase all text
    sentences = sentences.lower()

    # Remove punctuation and special characters
    punctuations = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
    for char in punctuations:
        sentences = sentences.replace(char, "")

    # Remove stop words (replace with your stop word list)
    # stop_words = ["a", "an", "the", "is", "of", "to"]
    # sentences = " ".join([word for word in sentences.split() if word not in stop_words])

    # Optional: Stemming or Lemmatization (implement if needed)
    # You can implement stemming or lemmatization using libraries like NLTK or spaCy

    sentiment = TextBlob(sentences).sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

    return sentences, sentiment or 0  # Return both preprocessed text and sentiment score

processed_data = []
for item in data:
    for i in range(0,3):
        sentence_tokens, sentence_sentiment_score = preprocess_text(item["sentences"]["sentence"][i])
        context_tokens, context_sentiment_score = preprocess_text(item["context"])
        processed_data.append({"sentence": sentence_tokens, "label": item["sentences"]["labels"][i]["label"], "sentence_sentiment_score": sentence_sentiment_score, "bias_type" : item["bias_type"], "context": context_tokens, "context_sentiment_score": context_sentiment_score})

# Extract text tokens from each data point
text_data = [item["sentence"] for item in processed_data]

context = [item["context"] for item in processed_data]

labels = [item["label"] for item in processed_data]

sentence_sentiment_scores = [item["sentence_sentiment_score"] for item in processed_data]

context_sentiment_scores = [item["context_sentiment_score"] for item in processed_data]

bias_type = [item["bias_type"] for item in processed_data]

train_size = 0.8  # 80% for training
test_size = 0.2   # 20% for testing

text_train, text_test, labels_train, labels_test, sentence_sentiment_score_train, sentence_sentiment_score_test, bias_type_train, bias_type_test, context_train, context_test, context_sentiment_score_train, context_sentiment_score_test = train_test_split(text_data, labels, sentence_sentiment_scores, bias_type, context, context_sentiment_scores,test_size=test_size, random_state=42)

# Initialize the Tokenizer
# tokenizer = keras_hub.tokenizers.Tokenizer()
tokenizer = keras.preprocessing.text.Tokenizer()

# Fit the tokenizer on the training text data (creates word indices)
tokenizer.fit_on_texts(text_train)

# Training Tokenized sequences (each word replaced with an integer)
tokenized_sequences_train = tokenizer.texts_to_sequences(text_train)

# Test Tokenized sequences (each word replaced with an integer)
tokenized_sequences_test = tokenizer.texts_to_sequences(text_test)

# Define the size of the vocabulary and embedding dimension
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
embedding_dim = 8  # Embedding dimension

# Pad the sequences to ensure uniform input shape
max_sequence_length = 100  # Define a fixed length for all sequences
padded_sequences_train = pad_sequences(tokenized_sequences_train, maxlen=max_sequence_length, padding='post')
padded_sequences_test = pad_sequences(tokenized_sequences_test, maxlen=max_sequence_length, padding='post')

# Sentence TF-IDF features
sentence_vectorizer_tfidf = TfidfVectorizer(max_features=1000)
sentence_tfidf_features_train = sentence_vectorizer_tfidf.fit_transform(text_train).toarray()
sentence_tfidf_features_test = sentence_vectorizer_tfidf.transform(text_test).toarray()

joblib.dump(sentence_vectorizer_tfidf, 'sentence_tfidf_vectorizer.pkl')

context_vectorizer_tfidf = TfidfVectorizer(max_features=1000)
context_tfidf_features_train = context_vectorizer_tfidf.fit_transform(context_train).toarray()
context_tfidf_features_test = context_vectorizer_tfidf.fit_transform(context_test).toarray()

joblib.dump(context_vectorizer_tfidf, 'context_tfidf_vectorizer.pkl')

np.array(bias_type_train).reshape(-1,1)
np.array(bias_type_test).reshape(-1,1)

np.array(sentence_sentiment_score_train).reshape(-1, 1)
np.array(sentence_sentiment_score_test).reshape(-1, 1)

np.array(context_sentiment_score_train).reshape(-1, 1)
np.array(context_sentiment_score_test).reshape(-1, 1)

bias_type_train_filtered = [t for t in bias_type_train if t is not None]  # Filter out None values
bias_type_test_filtered = [t for t in bias_type_test if t is not None]  # Filter out None values

# Encode bias types (assuming filtered bias_type):
le = LabelEncoder()
bias_type_train_filtered_encoded = le.fit_transform(bias_type_train_filtered)
bias_type_test_filtered_encoded = le.transform(bias_type_test_filtered)

joblib.dump(le, 'label_encoder.pkl')

sentence_sentiment_scores_train_filtered = [s or 0 for s in sentence_sentiment_score_train]  # Replace None with 0 (or a suitable default)
sentence_sentiment_scores_test_filtered = [s or 0 for s in sentence_sentiment_score_test]

context_sentiment_scores_train_filtered = [s or 0 for s in context_sentiment_score_train]  # Replace None with 0 (or a suitable default)
context_sentiment_scores_test_filtered = [s or 0 for s in context_sentiment_score_test]

# Loading of test datasets
joblib.dump(sentence_tfidf_features_test, 'sentence_tfidf_features_test.pkl')
joblib.dump(context_tfidf_features_test, 'context_tfidf_features_test.pkl')
joblib.dump(sentence_sentiment_scores_test_filtered, 'sentence_sentiment_scores_test_filtered.pkl')
joblib.dump(context_sentiment_scores_test_filtered, 'context_sentiment_scores_test_filtered.pkl')
joblib.dump(bias_type_test_filtered_encoded, 'bias_type_test_filtered_encoded.pkl')

np.array(sentence_sentiment_scores_train_filtered).reshape(-1, 1)  # Consider using pd.to_numpy
np.array(sentence_sentiment_scores_test_filtered).reshape(-1, 1)

# Sentence Input layers
sentences_tfidf_input = keras.Input(shape=(1000,))
sentence_sentiment_input = keras.Input(shape=(1,))

# Context Input layers
context_tfidf_input = keras.Input(shape=(1000,))
context_sentiment_input = keras.Input(shape=(1,))

# Sentence TF-IDF dense layer
sentence_tfidf_features_output = keras.layers.Dense(32, activation='relu')(sentences_tfidf_input)
sentence_tfidf_features_output = keras.layers.Dropout(0.1)(sentence_tfidf_features_output)

# Context TF-IDF dense layer
context_tfidf_features_output = keras.layers.Dense(32, activation='relu')(context_tfidf_input)
context_tfidf_features_output = keras.layers.Dropout(0.1)(context_tfidf_features_output)

# Concatenate layers
embedding_dim = 50  # Embedding dimension (adjust as needed)
embedding_input = keras.Input(shape=(max_sequence_length,))  # Input for padded sequences
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(embedding_input)
# embedding_output = keras.layers.GlobalAveragePooling1D()(embedding_layer)
lstm_output = keras.layers.LSTM(units=64, return_sequences=False)(embedding_layer)  # `return_sequences=True` if stacking LSTMs

merged = keras.layers.concatenate([sentence_tfidf_features_output, sentence_sentiment_input, context_tfidf_features_output, context_sentiment_input, lstm_output])

# Output layer
output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax')(merged)
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    num_units = trial.suggest_int('num_units', 32, 256, step=32)
    dropout_rate_1 = trial.suggest_float('dropout_rate_1', 0.1, 0.5)
    dropout_rate_2 = trial.suggest_float('dropout_rate_2', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    embedding_dim = 50  # Embedding dimension (adjust as needed)
    embedding_input = keras.Input(shape=(max_sequence_length,))  # Input for padded sequences
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(embedding_input)
    # embedding_output = keras.layers.GlobalAveragePooling1D()(embedding_layer)
    lstm_output = keras.layers.LSTM(units=num_units, return_sequences=False)(embedding_layer)  # `return_sequences=True` if stacking LSTMs

    # TF-IDF dense layer
    sentence_tfidf_features_output = keras.layers.Dense(num_units, activation='relu')(sentences_tfidf_input)
    sentence_tfidf_features_output = keras.layers.Dropout(dropout_rate_1)(sentence_tfidf_features_output)

    # Context TF-IDF dense layer
    context_tfidf_features_output = keras.layers.Dense(32, activation='relu')(context_tfidf_input)
    context_tfidf_features_output = keras.layers.Dropout(dropout_rate_2)(context_tfidf_features_output)

    # Concatenate layers
    merged = keras.layers.concatenate([sentence_tfidf_features_output,
                                       sentence_sentiment_input,
                                       context_tfidf_features_output,
                                       context_sentiment_input,
                                       lstm_output])

    # Output layer
    output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax')(merged)

    # Model compilation
    model = keras.Model(inputs=[sentences_tfidf_input, sentence_sentiment_input, context_tfidf_input, context_sentiment_input, embedding_input], outputs=output_layer)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    # Train the model
    model.fit(
        [np.array(sentence_tfidf_features_train),
         np.array(sentence_sentiment_scores_train_filtered).reshape(-1, 1),
         np.array(context_tfidf_features_train),
         np.array(context_sentiment_scores_train_filtered).reshape(-1,1), np.array(padded_sequences_train)],
        np.array(bias_type_train_filtered_encoded),
        epochs=10,
        batch_size=batch_size,
        verbose=0,
        validation_data=(
            [np.array(sentence_tfidf_features_test),
             np.array(sentence_sentiment_scores_test_filtered).reshape(-1, 1),
             np.array(context_tfidf_features_test),
             np.array(context_sentiment_scores_test_filtered).reshape(-1, 1),
             np.array(padded_sequences_test)],
            np.array(bias_type_test_filtered_encoded)
        )
    )

    # Evaluate the model
    _, val_accuracy = model.evaluate(
        [np.array(sentence_tfidf_features_test),
         np.array(sentence_sentiment_scores_test_filtered).reshape(-1, 1),
         np.array(context_tfidf_features_test),
         np.array(context_sentiment_scores_test_filtered).reshape(-1, 1),
         np.array(padded_sequences_test)],
        np.array(bias_type_test_filtered_encoded),
        verbose=0
    )

    return val_accuracy

study = optuna.create_study(direction='maximize')  # Maximize validation accuracy
study.optimize(objective, n_trials=3)  # Run 50 trials

print("Best hyperparameters:", study.best_params)

# Retrieve the best trial's hyperparameters
best_hyperparams = study.best_params

final_model = keras.Model(inputs=[sentences_tfidf_input, sentence_sentiment_input, context_tfidf_input, context_sentiment_input, embedding_input], outputs=output_layer)
final_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=best_hyperparams['learning_rate']),
    metrics=['accuracy']
)

final_model.fit(
    [np.array(sentence_tfidf_features_train),
     np.array(sentence_sentiment_scores_train_filtered).reshape(-1, 1),
     np.array(context_tfidf_features_train),
     np.array(context_sentiment_scores_train_filtered).reshape(-1,1),
     np.array(padded_sequences_train)],
    np.array(bias_type_train_filtered_encoded),
    epochs=10,
    batch_size=best_hyperparams['batch_size']
)

final_model.save('bias_type_model.keras')  # Save the model in the Keras format
