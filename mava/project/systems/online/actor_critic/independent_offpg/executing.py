from typing import Dict, Optional
from urllib.request import ProxyBasicAuthHandler

import tree
import dm_env
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt
from acme import types
from acme.tf import utils as tf2_utils

from mava.project.systems.online.value_based.independent_dqn import IndependentDQNExecutor

class IndependentOffPGExecutor(IndependentDQNExecutor):

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
            agents,
            q_network,
            action_selectors,
            variable_client,
            adder,
            evaluator
        )

        # We don't use the Q-network in the executor
        del self._q_network

        # Policy network setup during extra setup
        self._policy_network = None

    # HOOKS

    def extra_setup(self, **kwargs):
        self._policy_network = kwargs["policy_network"]

    def _reinitialize_core_states(self):
        # Re-initialize the recurrent core states with policy network.
        for agent in self._agents:
            self._core_states[agent] = self._policy_network.initial_state(1)

    def _select_action(self, agent, observation, legal_actions, core_state):

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.cast(tf.expand_dims(legal_actions, axis=0), "bool")

        # Pass observation through Q-network
        logits, next_core_state = self._policy_network(observation, core_state)

        # Sample action
        action = self._action_selectors[agent](logits, legal_actions)

        return action, next_core_state