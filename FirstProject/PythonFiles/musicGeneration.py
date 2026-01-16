import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm
assert len(tf.config.list_physical_devices('GPU')) == 0
songs = mdl.lab1.load_training_data()
example_song = songs[0]
# print("\nExample song: ")
# print(example_song)
# mdl.lab1.play_song(example_song)
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
# print("There are", len(vocab), "unique characters in the dataset.")

def vectorize_string(string):
    char2idx = {u: i for i, u in enumerate(string)}
    array = np.array(list(char2idx))
    return array
vectorized_songs = vectorize_string(songs_joined)
# print(vectorized_songs)
# print('{} ---- characters mapped to int ----> {}'.format(repr(songs_joined[:10]), vectorized_songs[:10]))
assert isinstance(vectorized_songs,np.ndarray), "returned result should be a numpy array" #"returned result should be a numpy array" is the assertion error message.
def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n-seq_length, batch_size)
    input_batch = list('bhavn')
    output_batch = list('havna')
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

test_args = (vectorized_songs, 5, 1)
if not mdl.lab1.test_batch_func_types(get_batch, test_args) or not mdl.lab1.test_batch_func_shapes(get_batch, test_args) or not mdl.lab1.test_batch_func_next_step(get_batch, test_args):
    print("======\n[FAIL] could not pass tests")
else:
    print("======\n[PASS] passed all tests!")
def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,  batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(rnn_units,activation=tf.keras.activations.softmax)
    ])
    return model
model = build_model(len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)
model.summary()
x, y = get_batch(vectorized_songs, seq_length=100, batch_size=32)
pred = model(x)
print("Input shape:      ", x.shape, "#(batch_size, sequence_length)")
print("Prediction shape: ", pred.shape, "#(batch_size, sequence_length, vocab_size)")
