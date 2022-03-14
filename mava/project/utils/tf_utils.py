import tensorflow as tf

def gather(values, indices, axis=-1, keepdims=False):
    one_hot_indices = tf.one_hot(indices, depth=values.shape[axis])
    if len(values.shape) > 4: # we have extra dim for distributional q-learning
        one_hot_indices = tf.expand_dims(one_hot_indices, axis=-1)
    gathered_values = tf.reduce_sum(values * one_hot_indices, axis=axis, keepdims=keepdims)
    return gathered_values

def quantile_regression_huber_loss(target, pred, num_atoms, tau):

    pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, num_atoms])
    target_tile = tf.tile(tf.expand_dims(target, axis=1), [1, num_atoms, 1])
    huber_loss = tf.compat.v1.losses.huber_loss(target_tile, pred_tile)
    tau = tf.cast(tf.reshape(tau, [1, num_atoms]), dtype='float32')
    inv_tau = 1.0 - tau
    tau = tf.tile(tf.expand_dims(tau, axis=1), [1, num_atoms, 1])
    inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, num_atoms, 1])
    error_loss = tf.math.subtract(target_tile, pred_tile)
    loss = tf.where(tf.less(error_loss, 0.0), inv_tau * huber_loss, tau * huber_loss)
    loss = tf.reduce_mean(tf.reduce_mean(loss, axis=-1), axis=-1)

    return loss