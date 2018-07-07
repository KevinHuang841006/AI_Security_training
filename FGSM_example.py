import keras
from keras.datasets import mnist
import numpy as np
from model import cnn_model
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir="/home/kevin/Desktop/FGSM"

#load_data():
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
nb_classes = 10
Y_train = keras.utils.to_categorical(Y_train, nb_classes)
Y_test  = keras.utils.to_categorical(Y_test, nb_classes)
X_train = np.reshape(X_train, (60000,28,28,1))
X_test = np.reshape(X_test, (10000,28,28,1))
#from sklearn import preprocessing
#X_train = preprocessing.scale(X_train)
#X_test = preprocessing.scale(X_test)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)    
print(Y_test.shape)

def test(x, preds, y, eps=0.3, ord=np.inf,clip_min=0, clip_max=1,targeted=False):
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    return y

def fgm(x, preds, y, eps=0.3, ord=np.inf,
        clip_min=0., clip_max=1.,
        targeted=False):
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keep_dims=True)
        y = tf.to_float(tf.equal(preds, preds_max))
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    #tf.Print(y)
    out = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)
    loss = tf.reduce_mean(out)
    
    grad, = tf.gradients(loss, x)
    normalized_grad = tf.sign(grad)
    scaled_grad = eps * normalized_grad
    
    adv_x = x + scaled_grad
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x
    

if __name__ == '__main__':
    print("start from here~")

    
    
    sess = tf.Session()
    
    model = cnn_model()
    
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input_data')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='output_layer')
    
    logit = model(x)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path
    #load model
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    print("Model loaded from: {}".format(ckpt_path))
    """
    sample = X_test[25:26]
    print(sample.shape)
    sess.run(logit, feed_dict={x: sample})
    """
    sample = X_test[25:26]
    
    one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
    target = 5
    one_hot_target[0, target] = 1
    label = one_hot_target
    #Testing~~~
    t = test(x,logit,y)
    print(sess.run(t, feed_dict={x: sample,y: label}))
    
    #FGSM
    adv_logit = fgm(x,logit,y)
    adv_img = sess.run(adv_logit, feed_dict={x: sample,y: label})  
    print(adv_img)
    adv_img = np.array(adv_img, dtype='float')
    pixels = adv_img.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    #from cleverhans.utils import pair_visual, grid_visual
    #grid_visual(np.reshape(adv_img, (1, 1 ,28, 28, 1)))
    
    
