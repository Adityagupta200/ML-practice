import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

sport  = tf.constant("Tennis", tf.string)
number = tf.constant([1.41421356237, 1.41421356237], tf.float64)

# print("'sport' is a {}-d Tensor".format(tf.rank(sport).numpy()))
# print("'number' is a {}-D Tensor".format(tf.rank(number).numpy()))

matrix = tf.constant([[1,2,3],[4,5,6]], tf.int8)
# print("'matrix' is a {}-D Tensor".format(tf.rank(matrix).numpy()))

assert isinstance(matrix, tf.Tensor)
assert tf.rank(matrix).numpy() == 2

images = tf.constant(tf.zeros((10,256,256,3),tf.int8))
# print("'images' is a {}-D Tensor".format(tf.rank(images).numpy()))

assert isinstance(images, tf.Tensor)
assert tf.rank(images).numpy() == 4
assert tf.shape(images).numpy().tolist() == [10,256,256,3]

row_vector = matrix[1]
column_vector = matrix[:, 1]
scalar = matrix[0,1]

# print("'row_vector' : {}".format(row_vector.numpy()))
# print("'coloumn_vector' : {}".format(column_vector.numpy()))
# print("'scalar' : {}".format(scalar.numpy()))

a = tf.constant(15)
b = tf.constant(61)

c1 = tf.add(a, b)
c2 = a + b
# print(c1)
# print(c2)

def func(a, b):
    c = tf.add(a, b)
    d = tf.subtract(b, 1)
    e = tf.multiply(c, d)
    return e

# print(func(4, 3))
# a, b = 1.5, 2.5
# e_out = func(a, b)
# print(e_out)

# class OurDenseLayer(tf.keras.layers.Layer):
#     def __init__(self, n_output_nodes):
#         super(OurDenseLayer, self).__init__()
#         self.n_output_nodes = n_output_nodes
#     def build(self, input_shape):
#         d = int(input_shape[-1])
#         self.W = self.add_weight("weight", shape=[d, self.n_output_nodes])
#         self.b = self.add_weight(name="bias", shape= [1, self.n_output_nodes])
#     def call(self, inputs, *args, **kwargs):
#         z = tf.matmul(inputs,self.W) + self.b
#         y = tf.sigmoid(z)
#         return y
# tf.random.set_seed(1)
# layer = OurDenseLayer(3)
# layer.build((1,2))
# x_input = tf.constant([[1,2.]],shape=(1,2))
# y = layer.call(x_input)
# # print(y.numpy())
#
# n_output_nodes = 3

# input_dense_layer = tf.keras.layers.Dense(units=32, activation='sigmoid', input_shape=(784,)),
# tf.keras.layers.Dense(units=32, activation='sigmoid', input_shape=(784,)),
# model = tf.keras.models.Sequential()
# dense_layer = tf.keras.layers.Dense(units=n_output_nodes, activation= 'sigmoid')
# model.add(dense_layer)
# x_input = tf.constant([[1,2.]], shape=(1,2))
# model_output = model.output
# print(model_output)
# print(tf.__version__)

# class SubclassModel(tf.keras.Model):
#     def __init__(self, n_output_nodes):
#         super(SubclassModel, self).__init__()
#         self.dense_layer = tf.keras.layers.Dense(units=n_output_nodes, activation=tf.nn.softmax)
#     def call(self, inputs, training=None, mask=None):
#         return self.dense_layer(inputs)
# n_output_nodes = 3
# model = SubclassModel(n_output_nodes)
# x_input = tf.constant([[1,2.]], shape= (1,2))
# print(model.call(x_input))

# class IdentityModel(tf.keras.Model):
#
#     def __init__(self, output_nodes):
#         super(IdentityModel,self).__init__()
#         self.dense_layer = tf.keras.layers.Dense(n_output_nodes,activation=tf.keras.activations.sigmoid)
#
#     def call(self, inputs, training=None, mask=None, isIdentity= True):
#         return self.dense_layer(inputs)
#
# output_nodes = 3
# identityModel = IdentityModel(output_nodes)
# identityModel_inputs = tf.constant([[2,3.]], shape=(1,2))
# print(identityModel.call(identityModel_inputs,None,None,True))
# print(identityModel.call(identityModel_inputs,None,None,False))
# x = tf.Variable(3.0)
# with tf.GradientTape() as tape:
#     y = x*x
#     dy_dx = tape.gradient(y,x)
#     assert dy_dx.numpy() == 6.0
x = tf.Variable([tf.random.normal([1])])
print("Initializing x = {}".format(x.numpy()))
learning_rate = 1e-2;
history = []
x_f = 4
for i in range(500):
    with tf.GradientTape() as tape:
        loss = tf.math.pow((x-x_f),2)
        grad = tape.gradient(loss, x)
        new_x = x - learning_rate * grad
        x.assign(new_x)
        history.append(x.numpy()[0])
plt.plot(history)
plt.plot((0,500), [x_f,x_f])
plt.legend(['Predicted', 'True'])
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()
