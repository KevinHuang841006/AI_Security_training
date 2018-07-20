import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2

input_height=28
input_width=28
tf_dtype = tf.as_dtype('float32')
np_dtype = np.dtype('float32')

"""
load_data():
"""

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.array(X_train, dtype='float32')
Y_train = np.array(Y_train, dtype='float32')
X_test = np.array(X_test, dtype='float32')
Y_test = np.array(Y_test, dtype='float32')

"""
Produce Adversarial data~~
"""

data = []
for i in range(2):
	tmp = cv2.resize(X_train[i], (56, 56),interpolation=cv2.INTER_NEAREST) 
	if i ==0 :
		for j in range(56):
			for k in range(56):
				if k % 2==1 or j % 2==1:
					tmp[j][k]=0
	if i ==1 :
		for j in range(56):
			for k in range(56):
				if k % 2==1 or j % 2==1:
					data[0][j][k] = tmp[j][k]
	data.append(tmp)
data = np.array(data)


"""
start here : down-scaling attack
data[0] : picture after mix '5' and '0'

Target Attack API : tensorflow -> tf.image.resize_nearest_neighbor

"nearest neighbor"

now after image tf.resize, the image will show '5'

"""

data = np.reshape(data,(2,56,56,1))

sample = np.reshape(data[0], (56,56))
plt.imshow(sample, cmap='gray')
plt.show()



sess = tf.Session()
init = tf.global_variables_initializer()

x = tf.placeholder(tf_dtype, shape=(None, 56, 56, 1), name='input_data')
# Target API
resized = tf.image.resize_nearest_neighbor(x, [input_height, input_width] )

sess.run(init)
ans = sess.run(resized, feed_dict={x: data })
print(len(ans))

ans = np.array(ans)


sample = np.reshape(ans[0],(28,28))
plt.imshow(sample, cmap='gray')
plt.show()

