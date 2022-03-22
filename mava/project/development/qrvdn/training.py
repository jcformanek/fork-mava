from typing import List, Dict, Optional, Any, Sequence

import tensorflow as tf
import sonnet as snt
import numpy as np
from acme.utils import loggers

from mava.project.systems.online.independent_qrdqn import IndependentQRDQNTrainer
from mava.project.utils.tf_utils import quantile_regression_huber_loss

class QRVDNTrainer(IndependentQRDQNTrainer):

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

    def _mixing(
        self, 
        chosen_action_qs, 
        target_max_qs,
        states
    ):
        """NoOp in independent DQN."""
        chosen_action_qs = tf.reduce_sum(chosen_action_qs, axis=-2)
        target_max_qs = tf.reduce_sum(target_max_qs, axis=-2)
        return chosen_action_qs, target_max_qs

    def _compute_targets(self, rewards, env_discounts, target_max_qs):
        targets = tf.stop_gradient(rewards + self._discount * env_discounts * target_max_qs)
        return targets

    def _compute_loss(self, chosen_actions_qs, targets):
        B = targets.shape[0]
        T = targets.shape[1]

        # Flatten batch and time dim
        pred = tf.reshape(chosen_actions_qs, shape=(-1, self._num_atoms))
        target = tf.reshape(targets, shape=(-1, self._num_atoms))

        # Quantile regression loss
        loss = quantile_regression_huber_loss(target, pred, self._num_atoms, self._tau)

        loss = tf.reshape(loss, shape=(B,T, 1))

        return loss