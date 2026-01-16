import keras.losses
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
print("Ran the import statements.")

train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv");
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean)/train_df_std
train_df_norm.head()

test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std
test_df_norm.head()

threshold = 265000
train_df_norm["median_house_value_is_high"] = (train_df["median_house_value"] > threshold).astype(float)
test_df_norm["median_house_value_is_high"] = (test_df["median_house_value"] > threshold).astype(float)
train_df_norm["median_house_value_is_high"].head(8000)

feature_columns = []
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

tr = tf.feature_column.numeric_column("total_rooms")
feature_columns.append(tr)
feature_layer = layers.DenseFeatures(feature_columns)
# feature_layer(dict(train_df_norm))

def create_model(my_learning_rate, feature_layer, my_metrics):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,), activation=tf.sigmoid),)
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=my_metrics)
    return model

def train_model(model, dataset, epochs, label_name, batch_size = None, shuffle = True):
    features = {name : np.array(value) for name, value in dataset.items()}
    # dataset = tf.data.Dataset.from_tensor_slices((features, label_name))
    # dataset = dataset.batch(batch_size).shuffle(shuffle)
    # history = model.fit(dataset, epochs=epochs)
    label = np.array(features.pop(label_name))
    history = model.fit(x= features, y= label, batch_size= batch_size, epochs= epochs, shuffle = shuffle)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist
print("Defined the create_model and train_model functions.")

def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label= m)

    plt.legend()
    plt.show()
print("Defined the plot_cuve function.")

learning_rate = 0.001
epochs = 20
batch_size = 100
label_name = "median_house_value_is_high"
classification_threshold = 0.35

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold= classification_threshold),
    tf.keras.metrics.Precision(name= 'precision', thresholds=classification_threshold),
    tf.keras.metrics.Recall(name= "recall", thresholds=classification_threshold),
    tf.keras.metrics.AUC(num_thresholds=100, name='auc')
]

my_model = create_model(learning_rate, feature_layer, METRICS)
epochs, hist = train_model(my_model, train_df_norm, epochs, label_name, batch_size)
list_of_metrics_to_plot = ['accuracy', 'precision', 'recall', 'auc']
plot_curve(epochs, hist, list_of_metrics_to_plot)
