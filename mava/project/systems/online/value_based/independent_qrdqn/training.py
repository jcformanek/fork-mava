from typing import List

import tensorflow as tf
import sonnet as snt
import numpy as np
from acme.utils import loggers

from mava.project.systems.online.value_based.independent_dqn import IndependentDQNTrainer
from mava.project.utils.tf_utils import gather, quantile_regression_huber_loss

class IndependentQRDQNTrainer(IndependentDQNTrainer):

    def __init__(
        self,
        agents: List[str],
        q_network: snt.Module,
        optimizer: snt.Optimizer,
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
    ):
        """Initialise Recurrent MADQN trainer

        Args:
            TODO
        """
        super().__init__(
            agents=agents,
            q_network=q_network,
            optimizer=optimizer,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
        )

    # HOOKS

    def extra_setup(self, **kwargs):
        self._num_atoms = kwargs["num_atoms"]
        self._tau = np.array([(2*(i-1)+1)/(2*self._num_atoms) for i in range(1, self._num_atoms+1)])

    def _get_chosen_action_qs(self, qs_out, actions):
        qs_out = tf.reshape(qs_out, shape=(*qs_out.shape[:-1], -1, self._num_atoms))
        return gather(qs_out, actions, axis=3, keepdims=False)

    def _get_target_max_qs(self, qs_out, target_qs_out, legal_actions):
        qs_out = tf.reshape(qs_out, shape=(*qs_out.shape[:-1], -1, self._num_atoms))
        target_qs_out = tf.reshape(target_qs_out, shape=(*target_qs_out.shape[:-1], -1, self._num_atoms))


        qs_out_selector = tf.reduce_mean(qs_out, axis=-1) # Mean of quantile distribution
        qs_out_selector = tf.where(legal_actions, qs_out_selector, -9999999) # legal action masking
        cur_max_actions = tf.argmax(qs_out_selector, axis=3)
        target_max_qs = gather(target_qs_out, cur_max_actions, axis=3)
        return target_max_qs

    def _compute_targets(self, rewards, env_discounts, target_max_qs):
        # We have extra distribution dim in target_max_qs
        # Expand dims to make compatible
        rewards = tf.expand_dims(rewards, axis=-1)
        env_discounts = tf.expand_dims(env_discounts, axis=-1)

        targets = tf.stop_gradient(rewards + self._discount * env_discounts * target_max_qs)
        return targets

    def _compute_loss(self, chosen_actions_qs, targets):
        B = targets.shape[0]
        T = targets.shape[1]

        loss_out = []
        for i in range(chosen_actions_qs.shape[2]): # loop over agents
            pred = chosen_actions_qs[:,:,i] # Get agents distribution
            target = targets[:,:,i] # Get agents distribution

            # Flatten batch and time dim
            pred = tf.reshape(pred, shape=(-1, self._num_atoms))
            target = tf.reshape(target, shape=(-1, self._num_atoms))

            # Quantile regression loss
            loss = quantile_regression_huber_loss(target, pred, self._num_atoms, self._tau)
        
            loss = tf.reshape(loss, shape=(B,T))

            loss_out.append(loss)

        loss_out = tf.stack(loss_out, axis=-1) # stack on agent dim

        return loss_out