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
