import numpy as np
import tensorflow as tf

class OurLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        pass

    def fn(self, inputs):
        output =  tf.pad(inputs[0], [[int(np.floor((inputs.shape[0] - 1) * inputs.shape[1] / 2)), int(np.ceil((inputs.shape[0] - 1) * inputs.shape[1] / 2))]])
        output = tf.reshape(output, shape=(1, *output.shape, 1))
        for n in range(1, inputs.shape[0]):
            filter = tf.reshape(inputs[n], shape=(*inputs[n].shape, 1, 1))
            output = tf.nn.conv1d(output, filters=filter, stride=1, padding='SAME')

        return output[0]

    def call(self, inputs): # inputs shape (batch, num_agents, num_bins)
        # return self.fn(inputs[0])
        return tf.map_fn(self.fn, inputs)

class BetterMixer(tf.keras.layers.Layer):

    def __init__(self, num_atoms, num_agents):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_agents = num_agents

    def build(self, input_shape):
        pass

    def fn(self, inputs):
        output =  tf.pad(inputs[0], [[int(np.floor((inputs.shape[0] - 1) * inputs.shape[1] / 2)), int(np.ceil((inputs.shape[0] - 1) * inputs.shape[1] / 2))]])
        output = tf.reshape(output, shape=(1, *output.shape, 1))
        for n in range(1, inputs.shape[0]):
            filter = tf.reshape(inputs[n], shape=(*inputs[n].shape, 1, 1))
            output = tf.nn.conv1d(output, filters=filter, stride=1, padding='SAME')

        return output[0]

    def call(self, inputs): # inputs shape (batch, num_agents, num_bins)
        FINAL_NUM_BINS = self.num_atoms * self.num_agents

        fft = tf.keras.layers.Lambda(lambda x: tf.signal.rfft(x, fft_length=[FINAL_NUM_BINS]))(inputs)
        conv_fft = tf.keras.layers.Lambda(lambda x: tf.math.reduce_prod(x, axis=1))(fft)
        conv = tf.keras.layers.Lambda(lambda x: tf.signal.irfft(x, fft_length=[FINAL_NUM_BINS]))(conv_fft)

        return conv

# BATCH_SIZE = 3
# AGENTS = 2
# NUM_BINS = 9

# layer = BetterMixer(NUM_BINS, AGENTS)

# inputs = np.zeros((BATCH_SIZE, AGENTS, NUM_BINS))
# inputs[:, :, 3:6] = 1 / 3
# theta = tf.Variable(1, dtype='float32')

# with tf.GradientTape() as tape:
#     inputs = tf.convert_to_tensor(inputs, dtype='float32') * theta
#     output = layer(inputs)

# gradients = tape.gradient(output, theta)

# print(gradients)
