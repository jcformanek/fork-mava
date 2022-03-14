from typing import List

import tensorflow as tf
import sonnet as snt
import numpy as np
from acme.utils import loggers

from mava.project.systems.online.value_based.independent_dqn import IndependentDQNTrainer

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