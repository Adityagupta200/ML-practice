import tensorflow as tf

# Create a vocabulary of words.
vocabulary = ['dog', 'cat', 'house', 'car', 'tree']

# Create a embedding layer with 100 dimensions.
embedding_layer = tf.keras.layers.Embedding(
    input_dim=len(vocabulary), output_dim=100
)

# Create a batch of input data.
input_data = tf.constant(['dog', 'cat', 'house', 'car', 'tree'])

# Pass the input data through the embedding layer.
embedded_data = embedding_layer.call(input_data)

# Print the embedded data.
print(embedded_data)