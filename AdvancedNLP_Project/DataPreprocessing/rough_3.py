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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
from transformers import BertTokenizer, TFBertModel
import nltk

# Ensure required resources are downloaded
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')  # Optional for better lemmatization support
nltk.download('averaged_perceptron_tagger')

# Check if wordnet is already downloaded
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# # instantiate a distribution strategy
# tf.tpu.experimental.initialize_tpu_system(tpu)
# tpu_strategy = tf.distribute.TPUStrategy(tpu)


with open("/kaggle/input/stereoset/StereoSet.json", "r") as f:
    data = json.load(f)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))  # Use a pre-defined set of English stopwords


def preprocess_text(sentences):  # Function definition with correct indentation

    # Lowercase all text
    sentences = sentences.lower()

    # Remove punctuation and special characters
    punctuations = "\"#$%&()*+-/:;<=>@[\]^_`{|}~"
    for char in punctuations:
        sentences = sentences.replace(char, "")

    # Remove stopwords
    tokens = word_tokenize(sentences)  # Tokenize the sentence first
    tokens = [word for word in tokens if word.lower() not in stop_words]  # Remove stopwords

    # POS tagging and lemmatization
    tagged_tokens = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens
    ]

    # Reconstruct preprocessed sentence
    preprocessed_text = " ".join(lemmatized_tokens)

    # Sentiment analysis
    sentiment = TextBlob(sentences).sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

    return preprocessed_text, sentiment or 0  # Return both preprocessed text and sentiment score


# Function to convert POS tags to WordNet format
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


processed_data = []
unprocessed_sentences = []
for item in data:
    for i in range(0, 3):
        unprocessed_sentences.append(item["sentences"]["sentence"][i])
        sentence_tokens, sentence_sentiment_score = preprocess_text(item["sentences"]["sentence"][i])
        context_tokens, context_sentiment_score = preprocess_text(item["context"])
        processed_data.append({"sentence": sentence_tokens, "label": item["sentences"]["labels"][i]["label"],
                               "sentence_sentiment_score": sentence_sentiment_score, "bias_type": item["bias_type"],
                               "context": context_tokens, "context_sentiment_score": context_sentiment_score})

# Extract text tokens from each data point
text_data = [item["sentence"] for item in processed_data]

context = [item["context"] for item in processed_data]

labels = [item["label"] for item in processed_data]

sentence_sentiment_scores = [item["sentence_sentiment_score"] for item in processed_data]

context_sentiment_scores = [item["context_sentiment_score"] for item in processed_data]

bias_type = [item["bias_type"] for item in processed_data]

train_size = 0.8  # 80% for training
test_size = 0.2  # 20% for testing

(text_tfidf_train, text_tfidf_test,
 labels_train, labels_test,
 sentence_sentiment_score_train, sentence_sentiment_score_test,
 bias_type_train, bias_type_test,
 context_train, context_test,
 context_sentiment_score_train, context_sentiment_score_test,
 unprocessed_sentences_train, unprocessed_sentences_test) = train_test_split(text_data, labels,
                                                                             sentence_sentiment_scores, bias_type,
                                                                             context, context_sentiment_scores,
                                                                             unprocessed_sentences, test_size=test_size,
                                                                             random_state=42)

# Initialize the Tokenizer
# tokenizer = keras_hub.tokenizers.Tokenizer()

# Load pre-trained BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased', gradient_checkpointing=True)

# Tokenize and encode text using BERT tokenizer
def tokenize_with_bert(sentences, max_len=100):
    encodings = bert_tokenizer(
        sentences,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='tf'
    )
    return encodings


bert_train_encodings = tokenize_with_bert(unprocessed_sentences_train, max_len=50)
bert_test_encodings = tokenize_with_bert(unprocessed_sentences_test, max_len=50)


# Generate BERT embeddings
def generate_bert_embeddings(encodings):
    outputs = bert_model(encodings['input_ids'], attention_mask=encodings['attention_mask'])
    # Use [CLS] token embedding
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
    return cls_embeddings


# Generate embeddings batch by batch
batch_size = 16  # Or even smaller if necessary
bert_train_embeddings = []
for i in range(0, len(unprocessed_sentences_train), batch_size):
    batch = {
        key: value[i:i + batch_size] for key, value in bert_train_encodings.items()
    }
    bert_train_embeddings.append(generate_bert_embeddings(batch))

bert_train_embeddings = tf.concat(bert_train_embeddings, axis=0)

# bert_train_embeddings = generate_bert_embeddings(bert_train_encodings)
# bert_test_embeddings = generate_bert_embeddings(bert_test_encodings)

# Generate embeddings batch by batch
batch_size = 16  # Or even smaller if necessary
bert_test_embeddings = []
for i in range(0, len(unprocessed_sentences_test), batch_size):
    batch = {
        key: value[i:i + batch_size] for key, value in bert_test_encodings.items()
    }
    bert_test_embeddings.append(generate_bert_embeddings(batch))

bert_test_embeddings = tf.concat(bert_test_embeddings, axis=0)

tokenizer = Tokenizer()

# Fit the tokenizer on the training text data (creates word indices)
tokenizer.fit_on_texts(unprocessed_sentences_train)

# Training Tokenized sequences (each word replaced with an integer)
tokenized_sequences_train = tokenizer.texts_to_sequences(unprocessed_sentences_train)

# Test Tokenized sequences (each word replaced with an integer)
tokenized_sequences_test = tokenizer.texts_to_sequences(unprocessed_sentences_test)

joblib.dump(tokenized_sequences_test, 'bert_test_embeddings.pkl')

# Define the size of the vocabulary and embedding dimension
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding
embedding_dim = 8  # Embedding dimension

# Pad the sequences to ensure uniform input shape
max_sequence_length = 100  # Define a fixed length for all sequences
padded_sequences_train = pad_sequences(tokenized_sequences_train, maxlen=max_sequence_length, padding='post')
padded_sequences_test = pad_sequences(tokenized_sequences_test, maxlen=max_sequence_length, padding='post')

# Sentence TF-IDF features
sentence_vectorizer_tfidf = TfidfVectorizer(max_features=1000)
sentence_tfidf_features_train = sentence_vectorizer_tfidf.fit_transform(text_tfidf_train).toarray()
sentence_tfidf_features_test = sentence_vectorizer_tfidf.transform(text_tfidf_test).toarray()

context_vectorizer_tfidf = TfidfVectorizer(max_features=1000)
context_tfidf_features_train = context_vectorizer_tfidf.fit_transform(context_train).toarray()
context_tfidf_features_test = context_vectorizer_tfidf.fit_transform(context_test).toarray()

# Reshaping arrays
np.array(bias_type_train).reshape(-1, 1)
np.array(bias_type_test).reshape(-1, 1)

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

joblib.dump(bias_type_test_filtered_encoded, 'label_encoder.pkl')

sentence_sentiment_scores_train_filtered = [s or 0 for s in
                                            sentence_sentiment_score_train]  # Replace None with 0 (or a suitable default)
sentence_sentiment_scores_test_filtered = [s or 0 for s in sentence_sentiment_score_test]

context_sentiment_scores_train_filtered = [s or 0 for s in
                                           context_sentiment_score_train]  # Replace None with 0 (or a suitable default)
context_sentiment_scores_test_filtered = [s or 0 for s in context_sentiment_score_test]

# Loading of test datasets
joblib.dump(sentence_tfidf_features_test, 'sentence_tfidf_features_test.pkl')
joblib.dump(context_tfidf_features_test, 'context_tfidf_features_test.pkl')
joblib.dump(sentence_sentiment_scores_test_filtered, 'sentence_sentiment_scores_test_filtered.pkl')
joblib.dump(context_sentiment_scores_test_filtered, 'context_sentiment_scores_test_filtered.pkl')
joblib.dump(bias_type_test_filtered_encoded, 'bias_type_test_filtered_encoded.pkl')
joblib.dump(bert_test_embeddings, 'bert_test_embeddings.pkl')
joblib.dump(padded_sequences_test, 'padded_sequences_test.pkl')

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

# LSTM embedding
embedding_dim = 50  # Embedding dimension (adjust as needed)
embedding_input = keras.Input(shape=(max_sequence_length,))  # Input for padded sequences
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(
    embedding_input)
# embedding_output = keras.layers.GlobalAveragePooling1D()(embedding_layer)
lstm_output = keras.layers.LSTM(units=64, return_sequences=False)(
    embedding_layer)  # `return_sequences=True` if stacking LSTMs

# BERT layers
bert_input = keras.Input(shape=(768,), name='bert_input')  # 768 is BERT hidden size
bert_dense_output = Dense(32, activation='relu')(bert_input)
bert_dense_output = keras.layers.Dropout(0.1)(bert_dense_output)

merged = keras.layers.concatenate([sentence_tfidf_features_output,
                                   sentence_sentiment_input,
                                   context_tfidf_features_output,
                                   context_sentiment_input,
                                   lstm_output,
                                   bert_dense_output])

# Output layer
output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax')(merged)


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

    num_units_sentence_tfidf = trial.suggest_int('num_units_sentence_tfidf', 32, 256, step=32)
    num_units_context_tfidf = trial.suggest_int('num_units_sentence_tfidf', 32, 256, step=32)
    num_units_bert = trial.suggest_int('num_units_bert', 32, 256, step=32)
    num_units_lstm = trial.suggest_int('num_units_lstm', 32, 256, step=32)

    dropout_rate_sentence_tfidf = trial.suggest_float('dropout_rate_sentence_tfidf', 0.1, 0.5)
    dropout_rate_context_tfidf = trial.suggest_float('dropout_rate_context_tfidf', 0.1, 0.5)
    dropout_rate_bert = trial.suggest_float('dropout_rate_bert', 0.1, 0.5)

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    embedding_dim = 50  # Embedding dimension (adjust as needed)
    embedding_input = keras.Input(shape=(max_sequence_length,))  # Input for padded sequences
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(
        embedding_input)
    # embedding_output = keras.layers.GlobalAveragePooling1D()(embedding_layer)
    lstm_output = keras.layers.LSTM(units=num_units_lstm, return_sequences=False)(
        embedding_layer)  # `return_sequences=True` if stacking LSTMs

    # TF-IDF dense layer
    sentence_tfidf_features_output = keras.layers.Dense(num_units_sentence_tfidf, activation='relu')(
        sentences_tfidf_input)
    sentence_tfidf_features_output = keras.layers.Dropout(dropout_rate_sentence_tfidf)(sentence_tfidf_features_output)

    # Context TF-IDF dense layer
    context_tfidf_features_output = keras.layers.Dense(num_units_context_tfidf, activation='relu')(context_tfidf_input)
    context_tfidf_features_output = keras.layers.Dropout(dropout_rate_context_tfidf)(context_tfidf_features_output)

    # BERT layers
    bert_input = keras.Input(shape=(768,), name='bert_input')  # 768 is BERT hidden size
    bert_dense_output = Dense(num_units_bert, activation='relu')(bert_input)
    bert_dense_output = keras.layers.Dropout(dropout_rate_bert)(bert_dense_output)

    # Concatenate layers
    merged = keras.layers.concatenate([sentence_tfidf_features_output,
                                       sentence_sentiment_input,
                                       context_tfidf_features_output,
                                       context_sentiment_input,
                                       lstm_output,
                                       bert_dense_output])

    # Output layer
    output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax')(merged)

    # with tpu_strategy.scope():
    # Model compilation
    model = keras.Model(inputs=[sentences_tfidf_input,
                                sentence_sentiment_input,
                                context_tfidf_input,
                                context_sentiment_input,
                                embedding_input,
                                bert_input], outputs=output_layer)
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
         np.array(context_sentiment_scores_train_filtered).reshape(-1, 1),
         np.array(padded_sequences_train),
         np.array(bert_train_embeddings)],
        np.array(bias_type_train_filtered_encoded),
        epochs=10,
        batch_size=batch_size,
        verbose=0,
        validation_data=(
            [np.array(sentence_tfidf_features_test),
             np.array(sentence_sentiment_scores_test_filtered).reshape(-1, 1),
             np.array(context_tfidf_features_test),
             np.array(context_sentiment_scores_test_filtered).reshape(-1, 1),
             np.array(padded_sequences_test),
             np.array(bert_test_embeddings)],
            np.array(bias_type_test_filtered_encoded)
        )
    )

    # Evaluate the model
    _, val_accuracy = model.evaluate(
        [np.array(sentence_tfidf_features_test),
         np.array(sentence_sentiment_scores_test_filtered).reshape(-1, 1),
         np.array(context_tfidf_features_test),
         np.array(context_sentiment_scores_test_filtered).reshape(-1, 1),
         np.array(padded_sequences_test),
         np.array(bert_test_embeddings)],
        np.array(bias_type_test_filtered_encoded),
        verbose=0
    )

    return val_accuracy


study = optuna.create_study(direction='maximize')  # Maximize validation accuracy
study.optimize(objective, n_trials=3)  # Run 50 trials

print("Best hyperparameters:", study.best_params)

# Retrieve the best trial's hyperparameters
best_hyperparams = study.best_params

# with tpu_strategy.scope():

final_model = keras.Model(inputs=[sentences_tfidf_input,
                                  sentence_sentiment_input,
                                  context_tfidf_input,
                                  context_sentiment_input,
                                  embedding_input,
                                  bert_input], outputs=output_layer)
final_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=best_hyperparams['learning_rate']),
    metrics=['accuracy']
)

final_model.fit(
    [np.array(sentence_tfidf_features_train),
     np.array(sentence_sentiment_scores_train_filtered).reshape(-1, 1),
     np.array(context_tfidf_features_train),
     np.array(context_sentiment_scores_train_filtered).reshape(-1, 1),
     np.array(padded_sequences_train),
     np.array(bert_train_embeddings)],
    np.array(bias_type_train_filtered_encoded),
    epochs=10,
    batch_size=best_hyperparams['batch_size']
)

final_model.save('bias_type_model.keras')  # Save the model in the Keras format