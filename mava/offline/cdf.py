import numpy as np
import tensorflow as tf

BATCH_SIZE = 3
AGENTS = 4
NUM_BINS = 9

inputs = np.zeros((BATCH_SIZE, AGENTS, NUM_BINS))
inputs[:, :, 3:6] = 1 / 3
inputs = tf.convert_to_tensor(inputs)

output = tf.pad(inputs[:, 0], [[0, 0], [int(np.floor((inputs.shape[1] - 1) * inputs.shape[2] / 2)), int(np.ceil((inputs.shape[1] - 1) * inputs.shape[2] / 2))]])
output = tf.reshape(output, shape=(1, *output.shape, 1))


# agent_dists = tf.nn.softmax(logits, axis=1)


conv1 = tf.nn.conv1d(tf.pad(p1, [[0, 0], [4, 5], [0, 0]]), filters=p2, stride=1, padding='SAME')

conv2 = tf.nn.conv1d(tf.pad(conv1, [[0, 0], [4, 5], [0, 0]]), filters=p3, stride=1, padding='SAME')

print(conv2)
