## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time
import cv2
import random

#os.environ['CUDA_VISIBLE_DEVICES']=''

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    new_inputs = []
    new_targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])


            new_inputs.append(data.test_data[start+i])
            new_targets.append(data.test_labels[start+i])


            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)
    new_inputs = np.array(new_inputs)
    new_targets = np.array(new_targets)

    return inputs, targets, new_inputs, new_targets


if __name__ == "__main__":
    with tf.Session() as sess:
        #data, model =  MNIST(), MNISTModel("models/mnist", sess)
        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        data = ImageNet()
        model = InceptionModel(sess)
        attack = CarliniL2(sess, model, batch_size=1, max_iterations=1000, confidence=0)
        #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)
        inputs, targets, new_inputs, new_targets = generate_data(data, samples=10, targeted=True,
                                        start=0, inception=True)
        
        
        
        """
        #total_attack = []        
        #target_all = np.zeros( (10,1008) )
        #D2 = new_inputs[9]
        #D2 = D2 + .5
        #D2 = D2 * 255
        #cv2.imwrite('C_W_data/' + str(555) + '.png', D2)
        new_inputs = new_inputs[0]
        print(new_inputs.shape)
        new_inputs = np.reshape(new_inputs, (299,299,3))
        #new_inputs = cv2.resize(new_inputs, (310,310), interpolation=cv2.INTER_CUBIC)
        #new_inputs = np.reshape(new_inputs, (1,310,310,3))
        print(new_targets.shape)
        print(new_inputs.shape)
        
        check1 = model.predict(new_inputs)
        A1 = sess.run(check1)
        print("predict id: ", np.argmax(A1[0]) )
        #for i in range(10):
        #for op in tf.get_default_graph().get_operations():
        #    print(str(op.name)) 
        """


        
        for i in range(10):
            #target_all.append(np.zeros( ( 1,1008 ) ) )
            target3 = np.zeros((1,1008))
            ttt = random.randint(0,500)
            print("target id: ",ttt)
            target3[0][ttt] = 1
            input3 = new_inputs[i:i+1,:,:,:]
            adv = attack.attack(input3, target3)
            data = np.array(adv)
            data = np.reshape(data,(299,299,3))
            data = data + .5
            data = data * 255
            from scipy.misc import imsave
            imsave('C_W_data/' + str(i+2000) + '.png',data)
            #cv2.imwrite('C_W_data/' + str(i+1000) + '.jpg',data)
        
        

    
        #timestart = time.time()
        #target = np.zeros((1,1008))
        #ttt = random.randint(0,500)
        #print("target id: ",ttt)
        #target[0][ ttt ] = 1
        #adv = attack.attack(inputs, target)
        #p_adv = np.reshape(adv,(299,299,3))

        """

        adv = attack.attack(inputs, target_all)

        total_attack.append(adv)
        data = adv
        data = np.array(data)
        data = data + .5
        data = data * 255
        cv2.imwrite('C_W_data/' + i + '.png',data)


        np.save('ten_image_without_pre.npy',total_attack)

        """

        """
        check1 = model.predict(adv)
        A1 = sess.run(check1)
        print("predict id: ", np.argmax(A1[0]) )
        
        tt = inputs
        tt = np.reshape(tt,(299,299,3))
        adv = np.reshape(adv,(299,299,3))

 
        data3 = adv
        data3 = data3 + .5
        data3 = data3 * 255       
        data3 = np.array(data3)
        data3 = np.reshape(data3,(299,299,3))
        #data3 = np.arctanh(data3)
        #print(data3.reshape)
        #print(image)
        cv2.imwrite('img_try.png',data3)

        data1 = tt
        data1 = data1 + .5
        data1 = data1 * 255       
        data1 = np.array(data1)
        cv2.imwrite('img_try2.png',data1)
        """
        #timeend = time.time()
        """
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            print("Adversarial:")
            show(adv[i])
            
            print("Classification:", model.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
        """
