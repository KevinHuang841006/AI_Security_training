# AI_Security_training

## Setup
run python ./C_W.py
env : anoconda, python-3.5, numpy, tensorflow

## Attack : Carlini and Wagner L2
Proposed by Carlini and Wagner. It is an iterative, white box attack.

paper reference : https://arxiv.org/abs/1608.04644

## Model
model data structure : Reference from Cleverhan. Definded in models.py, witch providing numerous hirachical tensor model.

### 3 layer CNN
'Conv2D0': <tf.Tensor 'add_7:0' shape=(?, 14, 14, 64) dtype=float32>, 
'ReLU1': <tf.Tensor 'Relu:0' shape=(?, 14, 14, 64) dtype=float32>, 
'Conv2D2': <tf.Tensor 'add_8:0' shape=(?, 5, 5, 128) dtype=float32>, 
'ReLU3': <tf.Tensor 'Relu_1:0' shape=(?, 5, 5, 128) dtype=float32>, 
'Conv2D4': <tf.Tensor 'add_9:0' shape=(?, 1, 1, 128) dtype=float32>, 
'ReLU5': <tf.Tensor 'Relu_2:0' shape=(?, 1, 1, 128) dtype=float32>, 
'Flatten6': <tf.Tensor 'Reshape:0' shape=(?, 128) dtype=float32>, 
'logits': <tf.Tensor 'add_10:0' shape=(?, 10) dtype=float32>, 
'probs': <tf.Tensor 'Softmax:0' shape=(?, 10) dtype=float32>
