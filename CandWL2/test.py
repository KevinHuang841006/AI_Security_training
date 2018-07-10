import keras
from keras.datasets import mnist
import numpy as np
from train_model import model_train

import tensorflow as tf
import matplotlib.pyplot as plt
from models import make_basic_cnn
from CarliniWagner import CarliniWagnerL2

train_dir="/home/kevin/Desktop/FGSM"
np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')

#load_data():
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.array(X_train, dtype='float32')
Y_train = np.array(Y_train, dtype='float32')
X_test = np.array(X_test, dtype='float32')
Y_test = np.array(Y_test, dtype='float32')
nb_classes = 10
Y_train = keras.utils.to_categorical(Y_train, nb_classes)
Y_test  = keras.utils.to_categorical(Y_test, nb_classes)
X_train = np.reshape(X_train, (60000,28,28,1))
X_test = np.reshape(X_test, (10000,28,28,1))
#from sklearn import preprocessing
#X_train = preprocessing.scale(X_train)
#X_test = preprocessing.scale(X_test)

X_train = (X_train - 0) / (1 - 0)
X_train = np.clip(X_train, 0, 1)
X_train = (X_train * 2) - 1
X_train = np.tanh(X_train )

#X_test = (X_test - 0) / (1 - 0)
#X_test = np.clip(X_test, 0, 1)
#X_test = (X_test * 2) - 1
#X_test = np.tanh(X_test )

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)    
print(Y_test.shape)

if __name__ == '__main__':
    print("start from here~")


    sess = tf.Session()
    sess_training = tf.Session()
    
    model = make_basic_cnn()
    
    x = tf.placeholder(tf_dtype, shape=(None, 28, 28, 1), name='input_data')
    y = tf.placeholder(tf_dtype, shape=(None, 10), name='output_layer')
    logit = model.get_logits(x)

    init = tf.global_variables_initializer()
    sess.run(init)
    sess_training.run(init)
    
    saver = tf.train.Saver()
    
    
    try:
        saver.restore(sess, "model/CW_model.ckpt")
        print("Load model sucessful~")
    except:
        print("Model was not loaded, training from scratch.")
        train_params = {
            'nb_epochs': 1,
            'batch_size': 1,
            'learning_rate': 0.001,
            'train_dir': "/home/kevin/Desktop/FGSM/temp/model",
            'filename': "CW_model.ckpt"
        }
        #model_train propagate logit layer (before softmax_layer)
        model_train(sess, x, y, logit, X_train, Y_train,
                    1,1,0.001,"/home/kevin/Desktop/FGSM/temp/model","CW_model.ckpt")
    
    sample = X_test[0:1]
    #sample = np.array(sample, dtype=np.float32)
    target = np.zeros((1, nb_classes), dtype=np.float32)
    target_label = 1
    target[0, target_label] = 1
    #target[1, target_label] = 1
    
    CWL2 = CarliniWagnerL2(sess, model, batch_size=1, confidence=0,
                 targeted=True, learning_rate=5e-3,
                 binary_search_steps=5, max_iterations=1000,
                 abort_early=False, initial_const=1e-2,
                 clip_min=0, clip_max=1, num_labels=10, shape=(28,28,1))
    
    ANS = CWL2.attack(sample, target)
    print("GOOD END~~~~~~~~~~~~~~~~~~~")