import tensorflow as tf

def generateModel():
    model = tf.keras.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32,filter_size= 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        # Second convolutional layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

        #Fully connected classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
