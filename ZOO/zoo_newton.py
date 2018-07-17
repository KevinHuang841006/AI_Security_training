"""
ZOO : Newton
dataset : MNIST



"""
import keras
from keras.datasets import mnist
import numpy as np
from train_model import model_train
import random

import tensorflow as tf
import matplotlib.pyplot as plt
from models import make_basic_cnn

train_dir="/home/kevin/Desktop/ZOO"
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

X_train =  np.tanh(X_train ) 

X_test =  np.tanh(X_test )

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)    
print(Y_test.shape)

break_point = False

def gradient_hessian(sess ,logit,base, sample , target , c):
    #initialize
    h = 0.0001
    e1 = np.zeros((1,28,28,1))
    i = random.randint(0,27)
    j = random.randint(0,27)
    e1[0,i,j,0] = 1
    
    #loss1
    aloss1 = sample + ( h * e1 ) - base
    aloss1 = np.square(aloss1)
    aloss1 = np.sqrt(aloss1)
    aloss1 = np.sum(aloss1)
    
    bloss1 = sample - ( h * e1 ) - base
    bloss1 = np.square(bloss1)
    bloss1 = np.sqrt(bloss1)
    bloss1 = np.sum(bloss1)
    
    oloss1 = sample - base
    oloss1 = np.square(oloss1)
    oloss1 = np.sqrt(oloss1)
    oloss1 = np.sum(oloss1)
    
    #loss2
    ascore = sess.run(logit, feed_dict={x: sample + ( h * e1)})
    atarget = ascore * target
    aother = ascore * ( (target * -1) + 1)
    atarget = np.sum(atarget)
    if atarget == 0:
        atarget = random.randint(1, 100) / 100000
    atarget = np.log(atarget)
    aother = np.max( aother )
    aother = np.log( aother )
    aloss2 = 0
    if aother - atarget > 0:
        aloss2 = aother - atarget
    
    bscore = sess.run(logit, feed_dict={x: sample - ( h * e1)})
    btarget = bscore * target
    bother = bscore * ( (target * -1) + 1)
    btarget = np.sum(btarget)
    if btarget == 0:
        btarget = random.randint(1, 100) / 100000
    btarget = np.log(btarget)
    bother = np.max( bother )
    bother = np.log( bother )
    bloss2 = 0
    if bother - btarget > 0:
        bloss2 = bother - btarget
    
    oscore = sess.run(logit, feed_dict={x: sample })
    otarget = oscore * target
    oother = oscore * ( (target * -1) + 1)
    otarget = np.sum(otarget)
    if otarget == 0:
        otarget = random.randint(1, 100) / 100000
    otarget = np.log(otarget)
    oother = np.max( oother )
    oother = np.log( oother )
    oloss2 = 0
    if oother - otarget > 0:
        oloss2 = oother - otarget
    
    func_plus = aloss1 + aloss2 * c
    func_minus = bloss1 + bloss2 * c
    func_orig = oloss1 + oloss2 * c
    
    g = ( func_plus - func_minus ) / (2*h)
    h = ( func_plus - 2 * func_orig + func_minus ) / (h*h)
    
    return g , h , i , j
    """
    l_rate = 0.03
    sample = sample - np.sign(max_grad) * e1 * l_rate
    sample = np.clip(sample, 0. , 1.)
    check_range(sample)
    return sample
    """

def check_range(sample):
    sample = sample.reshape((784))
    for a in sample:
        if a<0 or a>1:
            print("error~")
    
def plot_image(sample):
    sample = sample.reshape((28,28))
    plt.imshow(sample, cmap='gray')
    plt.show()

def Run_Stocastic_Gradient(sess ,logit, base , sample ,target):
    lower_bound = 0
    upper_bound = 1e10
    binary_size = 16
    
    best_L2 = 10000000000
    best_img = sample
    
    c = 0.01
    for l in range(binary_size):
        
        sucess_or_not = False
        
        ans = 0
        for i in range(3000):
            grad , hess , i , j = gradient_hessian(sess ,logit ,base ,sample ,target, c )
            learn_rate = 0.01
        
            if np.max(hess) <= 0 :
                m = -1 * grad * learn_rate 
            else:
                m = -1 * (grad / hess) * learn_rate
            mod = np.zeros((1,28,28,1))
            mod[0,i,j,0] = m
            L2 = sample - base
            L2 = L2 * L2
            L2 = np.sum(L2)
            L2 = np.sqrt(L2)
            print("L2: ", L2)

            sample+=mod
            sample = np.clip(sample, 0 , 1)
            print("c: ",c )
            print("modify: ",mod[0,i,j,0])
            ans = np.argmax( sess.run(logit, feed_dict={x: sample }) )
            print("ans: ",ans)
            if ans == np.argmax(target):
                sucess_or_not = True
                if L2 < best_L2:
                    best_L2 = L2
                    best_img = sample
                #plot_image(sample)
                #break
        # Sucess        
        if sucess_or_not == True:
            upper_bound = min( c , upper_bound )
            if upper_bound < 1e9:
                c = (upper_bound + lower_bound) / 2
        # Fail
        else:
            lower_bound = max( c , lower_bound )
            if upper_bound <1e9:
                c = (upper_bound + lower_bound) / 2
            else:
                c*=10
    print("best L2: ",best_L2)
    plot_image(best_img)


if __name__ == '__main__':
    print("start from here~")
    #print(X_train[0])
    
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
        saver.restore(sess, "model/zoo_model.ckpt")
        print("Load model sucessful~")
    except:
        print("Model was not loaded, training from scratch.")
        train_params = {
            'nb_epochs': 1,
            'batch_size': 1,
            'learning_rate': 0.001,
            'train_dir': "/home/kevin/Desktop/ZOO/model",
            'filename': "zoo_model.ckpt"
        }
        #model_train propagate logit layer (before softmax_layer)
        model_train(sess, x, y, logit, X_train, Y_train,
                    1,1,0.001,"/home/kevin/Desktop/ZOO/model","zoo_model.ckpt")
    logit=tf.nn.softmax(logit)
    
    sample = X_test[1:2]
    #sample = np.array(sample, dtype=np.float32)
    target = np.zeros((1, nb_classes), dtype=np.float32)
    target_label = 5
    target[0, target_label] = 1
    #print("sample target: ",Y_test[1:2])
    score = sess.run(logit, feed_dict={x: sample})
    #print( score )
    modify = np.ones( (1 ,28,28,1) ) * 0.3
    #print( ((target * -1 ) + 1)*score )
    
    Run_Stocastic_Gradient(sess ,logit,sample, sample ,target)
    
    
    print("GOOD END~~~~~~~~~~~~~~~~~~~")
