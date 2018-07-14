import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D
import tensorflow as tf
import numpy as np
import os

def model_loss(y, logit):
    out = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y)
    out = tf.reduce_mean(out)
    return out

def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

def model_train(sess, x, y, logit, X_train, Y_train, nb_epochs,
            batch_size,
            learning_rate,
            train_dir,
            filename, feed=None, var_list=None):
    #if not isinstance(args, dict):
    #    args = vars(args)

    loss = model_loss(y, logit)
    
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = train_step.minimize(loss, var_list=var_list)

    sess.run(tf.global_variables_initializer())
    for epoch in range(nb_epochs):
        # Compute number of batches
        nb_batches = len(X_train)
        
        # Indices to shuffle training set
        index_shuf = list(range(len(X_train)))
        rng = np.random.RandomState()
        rng.shuffle(index_shuf)
                
        for batch in range(nb_batches):
            if batch % 1000==0:
                print("batch: ",batch)
            # Compute batch start and end indices
            start, end = batch_indices( batch, len(X_train), batch_size)

            # Perform one training step
            feed_dict = {x: X_train[index_shuf[start:end]],
                         y: Y_train[index_shuf[start:end]]}
            if feed is not None:
                feed_dict.update(feed)
            sess.run(train_step, feed_dict=feed_dict)
        #assert end >= len(X_train)  # Check that all examples were used
    save_path = os.path.join(train_dir, filename)
    saver = tf.train.Saver()
    saver.save(sess, save_path)


def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
    """
    Defines the right convolutional layer according to the
    version of Keras that is installed.
    :param filters: (required integer) the dimensionality of the output
                    space (i.e. the number output of filters in the
                    convolution)
    :param kernel_shape: (required tuple or list of 2 integers) specifies
                         the strides of the convolution along the width and
                         height.
    :param padding: (required string) can be either 'valid' (no padding around
                    input or feature map) or 'same' (pad to ensure that the
                    output feature map size is identical to the layer input)
    :param input_shape: (optional) give input shape if this is the first
                        layer of the model
    :return: the Keras layer
    """
    
    if input_shape is not None:
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                      strides=strides, padding=padding,
                      input_shape=input_shape)
    else:
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                      strides=strides, padding=padding)




def cnn_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    layers = [conv_2d(nb_filters, (8, 8), (2, 2), "same",
                      input_shape=input_shape),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    if logits==True:
        logits_tensor = model(input_ph)
    #model.add(Activation('softmax'))

    if logits==True:
        print("ok~")
        return model, logits_tensor
    else:
        return model