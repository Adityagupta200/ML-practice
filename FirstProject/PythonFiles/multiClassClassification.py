import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
np.set_printoptions(linewidth=200)
print("Ran the import statements")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_normalised = x_train/255.0
x_test_normalised = x_test/255.0
def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("value")
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], m)
    plt.legend()
print("Loaded the plot_curve method")
def create_model(my_learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss='sparse_cateegorical_crossentropy',
                  metrics=['accuracy'])
    return model