import tensorflowjs as tfjs
import tensorflow as tf

# Load your saved Keras model
model = tf.keras.models.load_model('C:/Users/abc/Downloads/bias_predict_model.keras')

# Save the model in TensorFlow.js format
tfjs.converters.save_keras_model(model, 'C:/Users/abc/Downloads/tfjs_model')

