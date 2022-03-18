from typing import List

import trfl
import tensorflow as tf
import sonnet as snt
import numpy as np
from acme.utils import loggers

from mava.project.systems.online.independent_dqn import IndependentDQNTrainer

class VDNTrainer(IndependentDQNTrainer):

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
        chosen_action_qs = tf.reduce_sum(chosen_action_qs, axis=-1, keepdims=True)
        target_max_qs = tf.reduce_sum(target_max_qs, axis=-1, keepdims=True)
        return chosen_action_qs, target_max_qs

    def _compute_targets(self, rewards, env_discounts, target_max_qs):
        if self._lambda is not None:

            # Make time major for trfl implementation of Q(Lambda)
            rewards = tf.transpose(rewards, perm=[1,0,2])
            env_discounts = tf.transpose(env_discounts, perm=[1,0,2])
            target_max_qs = tf.transpose(target_max_qs, perm=[1,0,2])

            # Q(lambda)
            targets = trfl.multistep_forward_view(
                tf.squeeze(rewards),
                tf.squeeze(self._discount * env_discounts),
                tf.squeeze(target_max_qs),
                lambda_=self._lambda,
                back_prop=False
            )
            # Unpack agent dim again
            targets = tf.expand_dims(targets, axis=-1)

            # Make batch major again
            targets = tf.transpose(targets, perm=[1,0,2])
        else:
            targets = rewards + self._discount * env_discounts * target_max_qs
        return tf.stop_gradient(targets)