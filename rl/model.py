import tensorflow as tf
import tensorflow.contrib.keras as keras

class ThreesNN:
    def __init__(self, in_dim, hidden_dims):
        self.input_dim = in_dim
        self.hidden_dims = hidden_dims

    def build_model(self):
        self.nn = keras.models.Sequential()
        in_dim = self.input_dim
        for h in self.hidden_dims:
            self.nn.add(tf.keras.layers.Dense(h, input_shape=(in_dim,), activation=tf.nn.relu,
                                          kernel_initializer=tf.keras.initializers.glorot_uniform()))
            in_dim = h