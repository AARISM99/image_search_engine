import numpy as np
import tensorflow as tf

X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = tf.constant([[0.0, 1.0], [2.0, 3.0]])

#print(X)
#print(K)

h, w = K.shape

#print(h)
#print(w)

#print(X.shape[0])
#print(X.shape[1])

Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
#print("Y.numpy = " + str(Y.numpy))

i = 0
j = 1

res = tf.reduce_sum(X[i: i + h, j: j + w] * K)
#print("res = " + str(res))

Y[i,j].assign(res)
#print("Y = " + str(Y.numpy))

#print(X)
#print(X[i: i + h, j: j + w])

x = tf.Variable(4.0)

with tf.GradientTape() as tape:
    y = x**2
    # dy = 2x * dx
    dy_dx = tape.gradient(y, x)  # tape.gradient(y, x) = dy/dx
    print(dy_dx.numpy())
