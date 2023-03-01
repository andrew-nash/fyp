import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt


class TripleLinearAct(tf.keras.layers.Layer):
    '''
    Three step linear unit, that thresholds/flattens both large and small magnitude
    components

    Hyperparameters to optimise
    '''
    def __init__(self, init=None, trainBias=True, **kwargs):
        if isinstance(init,float) or isinstance(init,int):
            self.init = tf.constant_initializer(init)
        else:
            self.init = tf.keras.initializers.RandomUniform(minval=0.05, maxval=1, seed=None)
        self.trainBias = trainBias
        super(TripleLinearAct, self).__init__(**kwargs)

    def build(self, input_shape):
        self.thr = self.add_weight(shape     = (1,1,1), initializer=self.init,
                                    constraint=tf.keras.constraints.NonNeg(),
                                    trainable = self.trainBias,      name='threshold')
        super(TripleLinearAct, self).build(input_shape)
    def get_config(self):
        config = super().get_config()
        config.update({
            "trainBias": self.trainBias
        })
        return config
    def call(self, inputs):
        intermed = tf.math.maximum(inputs-self.thr, 0.05*(inputs-self.thr))-1*tf.math.maximum(-1.0*(inputs+self.thr), -0.05*(inputs+self.thr))
        return tf.math.maximum(0.01*(intermed)-1.0, tf.math.minimum(0.01*(intermed)+1.0, intermed))


class BiasRelu(tf.keras.layers.Layer):
    '''
    Three step linear unit, that thresholds/flattens both large and small magnitude
    components

    Hyperparameters to optimise
    '''
    def __init__(self, init=None, trainBias=True, **kwargs):
        if isinstance(init,float) or isinstance(init,int):
            self.init = tf.constant_initializer(init)
        else:
            self.init = tf.keras.initializers.RandomUniform(minval=0.05, maxval=1, seed=None)
        self.trainBias = trainBias
        super(BiasRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        self.thr = self.add_weight(shape     = (1,1,1), initializer=self.init,
                                    trainable = self.trainBias,      name='threshold')
        super(BiasRelu, self).build(input_shape)
    def get_config(self):
        config = super().get_config()
        config.update({
            "trainBias": self.trainBias
        })
        return config
    def call(self, inputs):
        return tf.nn.relu(inputs+self.thr)