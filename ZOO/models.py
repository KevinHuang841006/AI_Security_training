"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.

cnn_basic_model_layers~~~~

'Conv2D0': <tf.Tensor 'add_7:0' shape=(?, 14, 14, 64) dtype=float32>, 
'ReLU1': <tf.Tensor 'Relu:0' shape=(?, 14, 14, 64) dtype=float32>, 
'Conv2D2': <tf.Tensor 'add_8:0' shape=(?, 5, 5, 128) dtype=float32>, 
'ReLU3': <tf.Tensor 'Relu_1:0' shape=(?, 5, 5, 128) dtype=float32>, 
'Conv2D4': <tf.Tensor 'add_9:0' shape=(?, 1, 1, 128) dtype=float32>, 
'ReLU5': <tf.Tensor 'Relu_2:0' shape=(?, 1, 1, 128) dtype=float32>, 
'Flatten6': <tf.Tensor 'Reshape:0' shape=(?, 128) dtype=float32>, 
'logits': <tf.Tensor 'add_10:0' shape=(?, 10) dtype=float32>, 
'probs': <tf.Tensor 'Softmax:0' shape=(?, 10) dtype=float32>
"""
from abc import ABCMeta
import numpy as np
import tensorflow as tf
#from model import Model

class Model(object):

    """
    An abstract interface for model wrappers that exposes model symbols
    needed for making an attack. This abstraction removes the dependency on
    any specific neural network package (e.g. Keras) from the core
    code of CleverHans. It can also simplify exposing the hidden features of a
    model when a specific package does not directly expose them.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """
        For compatibility with functions used as model definitions (taking
        an input tensor and returning the tensor giving the output
        of the model on that input).
        """
        return self.get_probs(*args, **kwargs)

    def get_layer(self, x, layer):
        """
        Expose the hidden features of a model given a layer name.
        :param x: A symbolic representation (Tensor) of the network input
        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation (Tensor) of the hidden features
        :raise: NoSuchLayerError if `layer` is not in the model.
        """
        # Return the symbolic representation (All Tensors~~~~~) for this layer.
        output = self.fprop(x)
        #print(output)
        
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        #print("requested: ",requested)
        return requested

    def get_logits(self, x):
        print("getting logits~~")
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the output logits
        (i.e., the values fed as inputs to the softmax layer).
        """
        return self.get_layer(x, 'logits')

    def get_probs(self, x):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the output
        probabilities (i.e., the output values produced by the softmax layer).
        """
        try:
            return self.get_layer(x, 'probs')
        except NoSuchLayerError:
            pass
        except NotImplementedError:
            pass
        import tensorflow as tf
        return tf.nn.softmax(self.get_logits(x))

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model abstraction.
        """

        if hasattr(self, 'layer_names'):
            return self.layer_names

        raise NotImplementedError('`get_layer_names` not implemented.')

    def fprop(self, x):
        """
        Exposes all the layers of the model returned by get_layer_names.
        :param x: A symbolic representation (Tensor) of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        raise NotImplementedError('`fprop` not implemented.')

    def get_params(self):
        """
        Provides access to the model's parameters.
        :return: A list of all Variables defining the model parameters.
        """
        raise NotImplementedError()

class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        # layer : class
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        states = []
        print("Start layers: ")
        for layer in self.layers:
            print(layer)
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        print("End layers~~~")
		#   (    tensor_name   :   tensor_object       )  
        states = dict(zip(self.get_layer_names(), states))
        return states

    def get_params(self):
        out = []
        for layer in self.layers:
            for param in layer.get_params():
                if param not in out:
                    out.append(param)
        return out


class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):

    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b

    def get_params(self):
        return [self.W, self.b]


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b

    def get_params(self):
        return [self.kernels, self.b]


class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.relu(x)

    def get_params(self):
        return []


class Softmax(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)

    def get_params(self):
        return []


class Flatten(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [shape[0], output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])

    def get_params(self):
        return []


def make_basic_cnn(nb_filters=64, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model
