# import json
# import keras
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from textblob import TextBlob
# import numpy as np
# from textblob import TextBlob
# import tensorflow as tf
# from collections import Counter
# from sklearn.preprocessing import LabelEncoder
# import optuna
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import accuracy_score
#
# # class ReshapeLayer(keras.layers.Layer):
# #   def __init__(self, target_shape, **kwargs):
# #     self.target_shape = target_shape
# #     super(ReshapeLayer, self).__init__(**kwargs)
# #
# #   def call(self, inputs):
# #     return tf.reshape(inputs, self.target_shape)
#
# # class SentimentEmbeddingLayer(keras.layers.Layer):
# #   def __init__(self, embedding_dim=32, units=64, activation='relu', **kwargs):
# #     self.embedding_dim = embedding_dim
# #     self.units = units
# #     self.activation = activation
# #     super(SentimentEmbeddingLayer, self).__init__(**kwargs)
# #
# #   def call(self, inputs):
# #     embedding = keras.layers.Embedding(input_dim=2308, output_dim=self.embedding_dim)(inputs)
# #     output = keras.layers.Dense(self.units, activation=self.activation)(embedding)
# #     return tf.reshape(output, (-1, self.units))  # Reshape here
#
# # Replace 'path/to/your/file.json' with the actual path to your StereoSet data file
# with open("C:/Users/abc/Downloads/StereoSet.json", "r") as f:
#   data = json.load(f)
#
# def preprocess_text(sentences):  # Function definition with correct indentation
#     """
#     This function preprocesses text data for bias detection.
#
#     Args:
#         sentences: A string containing the text to be preprocessed.
#
#     Returns:
#         A list of preprocessed tokens.
#     """
#
#     # Lowercase all text
#     sentences = sentences.lower()
#
#     # Remove punctuation and special characters
#     punctuations = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
#     for char in punctuations:
#         sentences = sentences.replace(char, "")
#
#     # Remove stop words (replace with your stop word list)
#     stop_words = ["a", "an", "the", "is", "of", "to"]
#     sentences = " ".join([word for word in sentences.split() if word not in stop_words])
#
#     # Optional: Stemming or Lemmatization (implement if needed)
#     # You can implement stemming or lemmatization using libraries like NLTK or spaCy
#
#     sentiment = TextBlob(sentences).sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)
#
#     return sentences, sentiment or 0  # Return both preprocessed text and sentiment score
#
# processed_data = []
# for item in data:
#     for i in range(0,3):
#         tokens, sentiment_score = preprocess_text(item["sentences"]["sentence"][i])
#         processed_data.append({"sentence": tokens, "label": item["sentences"]["labels"][i]["label"], "sentiment_score": sentiment_score, "bias_type" : item["bias_type"]})
#
# # Extract text tokens from each data point
# text_data = [item["sentence"] for item in processed_data]
#
# # BoW implementation
# # # Create a CountVectorizer object
# # vectorizer_bow = CountVectorizer()
# #
# # # Transform text data into BoW features (word counts)
# # bow_features = vectorizer_bow.fit_transform(text_data)
#
# # Create a TfidfVectorizer object with maximum features set to 1000
# # (you can adjust this parameter as needed)
# vectorizer_tfidf = TfidfVectorizer(max_features=1000)
#
# # Transform text data into TF-IDF features
# tfidf_features = vectorizer_tfidf.fit_transform(text_data)
#
# # This will create a sparse matrix, but the values represent the TF-IDF score
# # for each word in each document. TF-IDF considers both word frequency within
# # a document and its overall frequency across the corpus.
# # Example: Splitting 80% for training and 20% for testing
# train_size = 0.8  # 80% for training
# test_size = 0.2   # 20% for testing
#
# labels = [item["label"] for item in processed_data]
#
# # Extract sentiment scores as a separate list
# sentiment_scores = [item["sentiment_score"] for item in processed_data]
#
# bias_type = [item["bias_type"] for item in processed_data]
#
# # for i in tfidf_features:
# #     if (i == None):
# #         print(tfidf_features.index(i))
#
#
# # Ensure all arrays have the same number of elements (check lengths and trim/filter if needed)
# # data_length = len(labels)  # Assuming labels represent the number of samples
# # if len(text_data) != data_length:
# #     # Handle potential length mismatch in text_data (e.g., filter or trim)
# #     text_data = text_data[:data_length]  # Example: Truncate text_data
# #
# # if len(tfidf_features) != data_length:
# #     # Handle potential length mismatch in tfidf_features (e.g., filter or trim)
# #     tfidf_features = tfidf_features[:data_length]  # Example: Truncate tfidf_features
# #
# # if len(sentiment_scores) != data_length:
# #     # Handle potential length mismatch in sentiment_scores (e.g., filter or trim)
# #     sentiment_scores = sentiment_scores[:data_length]  # Example: Truncate sentiment_scores
#
# # Splitting text data (text_data) and labels (labels) into training and testing sets
# text_train, text_test, labels_train, labels_test, sentiment_score_train, sentiment_score_test, bias_type_train, bias_type_test = train_test_split(text_data, labels, sentiment_scores, bias_type, test_size=test_size, random_state=42)
# # Combine features (TF-IDF and sentiment score)
#
# # print(np.array(text_train).reshape(-1,1))
# # print(tfidf_features_train.reshape(-1,1))
# # print(np.array(sentiment_score_train).reshape(-1, 1).size)
#
# tfidf_features_train = vectorizer_tfidf.fit_transform(text_train).toarray()
# tfidf_features_test = vectorizer_tfidf.transform(text_test).toarray()
#
# all_features_train = np.concatenate((np.array(bias_type_train).reshape(-1,1), np.array(sentiment_score_train).reshape(-1, 1)), axis=1)
# all_features_test = np.concatenate((np.array(text_test).reshape(-1,1), np.array(bias_type_test).reshape(-1,1), np.array(sentiment_score_test).reshape(-1, 1)), axis=1)
#
# # Define the model architecture with separate input layers
# # model = keras.Sequential()
# #
# # keras.Input(shape=(tfidf_features.shape[1],)),  # Shape based on TF-IDF features
# #
# # keras.layers.Dense(128, activation='relu')(model.output)  # Example with 128 units and ReLU activation
# #
# # keras.Input(shape=(2,)),  # Assuming 3 features (sentences, sentiment score and bias type label)
# #
# # all_features_train_output = tf.keras.layers.Dense(64, activation='relu')(all_features_train)  # Example with 64 units and ReLU activation
# #
# # embedding_dim = 32
# #
# # # Embedding layer for all_features
# # all_features_input = keras.Input(shape=(3,))  # Assuming 3 features (sentence, sentiment score and bias type label)
# # embedded_all_features = keras.layers.Embedding(input_dim=2308, output_dim=embedding_dim)(all_features_train)
# #
# # # ... (rest of your code defining separate input layers for TF-IDF and embedded all_features)
# #
# # # Now use embedded_all_features instead of all_features_train/test in concatenation
# # merged = keras.layers.concatenate([model.output, embedded_all_features])
# #
# # # Output layer for prediction (e.g., binary classification for bias)
# # keras.layers.Dense(1, activation='sigmoid')  # Adjust for your task (e.g., multi-class)
# #
# # # Compile the model
# # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# #
# # # Train the model
# # model.fit([tfidf_features_train, all_features_train], epochs=10, validation_data=([tfidf_features_test, all_features_test]))
# #
# # # Make predictions on unseen text (assuming you have preprocessed unseen_text)
# # unseen_text_sequences = tf.keras.preprocessing.text.text_to_sequences(unseen_text)
# # predictions = model.predict([unseen_text_sequences, [[unseen_text_sentiment_score, unseen_text_bias_type]]])
#
# # Define the model architecture with separate input layers
#
# # model = keras.Sequential()
# #
# # # Input layer for TF-IDF features
# # tfidf_input = keras.Input(shape=(1,))  # Shape based on TF-IDF features
# #
# # # Input layer for sentiment score and bias type (combined)
# # sentiment_input = keras.Input(shape=(2,)) # Assuming 2 features (sentiment score and bias type label)
# #
# # # Hidden layer for processing TF-IDF features
# # tfidf_features_output = keras.layers.Dense(64, activation='relu', input_shape=(1,))(tfidf_input)  # Example with 128 units and ReLU activation
# # model.add(tfidf_features_output)
# #
# # # Embedding layer for sentiment score and bias type (optional, adjust input_dim)
# # # sentiment_bias_embedding = keras.layers.Embedding(input_dim=2308, output_dim=32)(sentiment_input)  # Adjust input_dim based on number of unique sentiment/bias type combinations
# #
# # # Hidden layer for processing sentiment and bias type embedding
# # # sentiment_bias_output = keras.layers.Dense(64, activation='relu')(sentiment_input)  # Example with 64 units and ReLU activation
# #
# # # Reshape layer with target shape (None, 64)
# # # reshape_layer = ReshapeLayer(target_shape=(-1, 64))
# # # sentiment_bias_output_reshaped = reshape_layer(sentiment_bias_output)
# # sentiment_layer = SentimentEmbeddingLayer(embedding_dim=32, units=64, input_shape=(2,))(sentiment_input)
# # model.add(sentiment_layer)
# #
# # # Proceed with concatenating the output with other features
# # merged = keras.layers.concatenate([tfidf_features_output, sentiment_layer])
# # # Additional hidden layers (optional)
# # # ... (add more hidden layers if needed)
# #
# # # Output layer for predicting bias type (multi-class classification)
# # output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax', input_shape=(3,))(merged)  # Adjust for number of bias types
# #
# # model.add(output_layer)
#
# # Compile the model
# bias_type_train_filtered = [t for t in bias_type_train if t is not None]  # Filter out None values
# bias_type_test_filtered = [t for t in bias_type_test if t is not None]  # Filter out None values
#
# # Encode bias types (assuming filtered bias_type):
# le = LabelEncoder()
# bias_type_train_filtered_encoded = le.fit_transform(bias_type_train_filtered)
# bias_type_test_filtered_encoded = le.fit_transform(bias_type_test_filtered)
#
# sentiment_scores_train_filtered = [s or 0 for s in sentiment_score_train]  # Replace None with 0 (or a suitable default)
# sentiment_scores_test_filtered = [s or 0 for s in sentiment_score_test]
#
# sentiment_score_train_array = np.array(sentiment_scores_train_filtered).reshape(-1, 1)  # Consider using pd.to_numpy
# sentiment_score_test_array = np.array(sentiment_scores_test_filtered).reshape(-1, 1)
#
# # sentiment_bias_train_combined = np.concatenate((np.array(sentiment_scores_train_filtered).reshape(-1,1), np.array(bias_type_train_filtered_encoded).reshape(-1,1)), axis=1)
# sentiment_test_combined = np.array(sentiment_scores_test_filtered).reshape(-1, 1)
#
# sentiment_train_tfidf_combined = np.concatenate([np.array(tfidf_features_train), np.array(sentiment_scores_train_filtered).reshape(-1, 1)], axis=1)
# sentiment_test_tfidf_combined = np.concatenate([np.array(tfidf_features_test), sentiment_test_combined], axis=1)
#
# # Model architecture:
#
# # Input layers:
# # tfidf_input = keras.Input(shape=(1000,))  # Input for TF-IDF features
# # sentiment_input = keras.Input(shape=(1,))  # Input for sentiment score and bias type
# #
# # # Hidden layers
# # tfidf_features_output = keras.layers.Dense(64,activation='relu')(tfidf_input)
# # # sentiment_layer = SentimentEmbeddingLayer(embedding_dim=32, units=64)(sentiment_input)
# #
# # # Concatenation
# # merged = keras.layers.concatenate([tfidf_features_output, sentiment_input])
# #
# # # Output layer
# # output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax')(merged)
# #
# # # Model definition
# # model = keras.Model(inputs=[tfidf_input, sentiment_input], outputs=output_layer)
# #
# # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# #
# # # predictions = model.predict(sentiment_test_tfidf_combined)
# # # predicted_labels = np.argmax(predictions, axis=1)
# # # accuracy = accuracy_score(bias_type_test_filtered_encoded, predicted_labels)
# #
# # model.fit(
# #     [np.array(tfidf_features_train), np.array(sentiment_scores_train_filtered).reshape(-1, 1)],  # List of inputs
# #     np.array(bias_type_train_filtered_encoded),  # Target labels
# #     epochs=10,
# #     batch_size=32,
# #     verbose=0,
# #     validation_data=(
# #         [np.array(tfidf_features_test), np.array(sentiment_scores_test_filtered).reshape(-1, 1)],  # Validation inputs
# #         np.array(bias_type_test_filtered_encoded)  # Validation labels
# #     )
# # )
#
# # Define model architecture
#
# #Input layers
# tfidf_input = keras.Input(shape=(1000,))
# sentiment_input = keras.Input(shape=(1,))
#
# # TF-IDF dense layer
# tfidf_features_output = keras.layers.Dense(32, activation='relu')(tfidf_input)
# tfidf_features_output = keras.layers.Dropout(0.1)(tfidf_features_output)
#
# # Concatenate layers
# merged = keras.layers.concatenate([tfidf_features_output, sentiment_input])
#
# # Output layer
# output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax')(merged)
#
# def objective(trial):
#     # Suggest hyperparameters
#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
#     num_units = trial.suggest_int('num_units', 32, 256, step=32)
#     dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
#     batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
#
#     # TF-IDF dense layer
#     tfidf_features_output = keras.layers.Dense(num_units, activation='relu')(tfidf_input)
#     tfidf_features_output = keras.layers.Dropout(dropout_rate)(tfidf_features_output)
#
#     # Concatenate layers
#     merged = keras.layers.concatenate([tfidf_features_output, sentiment_input])
#
#     # Output layer
#     output_layer = keras.layers.Dense(len(set(bias_type)), activation='softmax')(merged)
#
#     # Model compilation
#     model = keras.Model(inputs=[tfidf_input, sentiment_input], outputs=output_layer)
#     model.compile(
#         loss='sparse_categorical_crossentropy',
#         optimizer=Adam(learning_rate=learning_rate),
#         metrics=['accuracy']
#     )
#
#     # Train the model
#     model.fit(
#         [np.array(tfidf_features_train), np.array(sentiment_scores_train_filtered).reshape(-1, 1)],
#         np.array(bias_type_train_filtered_encoded),
#         epochs=10,
#         batch_size=batch_size,
#         verbose=0,
#         validation_data=(
#             [np.array(tfidf_features_test), np.array(sentiment_scores_test_filtered).reshape(-1, 1)],
#             np.array(bias_type_test_filtered_encoded)
#         )
#     )
#
#     # Evaluate the model
#     _, val_accuracy = model.evaluate(
#         [np.array(tfidf_features_test), np.array(sentiment_scores_test_filtered).reshape(-1, 1)],
#         np.array(bias_type_test_filtered_encoded),
#         verbose=0
#     )
#
#     return val_accuracy
#
# study = optuna.create_study(direction='maximize')  # Maximize validation accuracy
# study.optimize(objective, n_trials=3)  # Run 50 trials
#
# print("Best hyperparameters:", study.best_params)
#
# # Retrieve the best trial's hyperparameters
# best_hyperparams = study.best_params
#
# final_model = keras.Model(inputs=[tfidf_input, sentiment_input], outputs=output_layer)
# final_model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=Adam(learning_rate=best_hyperparams['learning_rate']),
#     metrics=['accuracy']
# )
#
# final_model.fit(
#     [np.array(tfidf_features_train), np.array(sentiment_scores_train_filtered).reshape(-1, 1)],
#     np.array(bias_type_train_filtered_encoded),
#     epochs=10,
#     batch_size=best_hyperparams['batch_size']
# )
#
# final_model.save('bias_type_model.keras')  # Save the model in the Keras format
#
