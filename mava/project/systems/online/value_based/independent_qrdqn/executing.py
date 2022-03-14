from typing import Dict, Optional

import tree
import dm_env
import numpy as np
import tensorflow as tf
import sonnet as snt
from acme import types
from acme.tf import utils as tf2_utils

from mava.project.systems.online.value_based.independent_dqn import IndependentDQNExecutor

class IndependentQRDQNExecutor(IndependentDQNExecutor):

    def __init__(
        self,
        agents,
        q_network: snt.Module,
        action_selectors: Dict,
        variable_client = None,
        adder = None,
        evaluator = False
    ):
        super().__init__(
            agents=agents,
            q_network=q_network,
            action_selectors=action_selectors,
            variable_client=variable_client,
            adder=adder,
            evaluator=evaluator 
        )

    # HOOKS

    def _select_action(self, agent, observation, legal_actions, core_state):
        num_actions = legal_actions.shape[-1]

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        # Pass observation through Q-network
        probs, next_core_state = self._q_network(observation, core_state)

        # Mean of distributions
        probs = tf.reshape(probs, shape=(1, num_actions,-1))
        action_values = tf.reduce_mean(probs, axis=-1)

        # Pass action values through action selector
        action = self._action_selectors[agent](action_values, legal_actions)

        return action, next_core_state