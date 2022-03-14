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

class IndependentOffPGExecutor:

    def __init__(
        self,
        agents,
        policy_network: snt.Module,
        variable_client = None,
        adder = None,
        evaluator = False
    ):
        self._agents = agents
        self._policy_network = policy_network
        self._variable_client = variable_client
        self._adder = adder

        self._evaluator = evaluator
        self._interval = None

        # Recurrent network core states
        self._core_states = {}

    def observe_first(
        self, 
        timestep: dm_env.TimeStep,
        extras: Dict = {}
    ):
        # Re-initialize the recurrent core states.
        for agent in timestep.observation.keys():
            self._core_states[agent] = self._policy_network.initial_state(1)

        if self._adder is not None:

            extras.update(
                {"zero_padding_mask": np.array(1)}
            )

            # Adder
            self._adder.add_first(timestep, extras)


    def observe(
        self,   
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record observed timestep from the environment.

        Args:
            actions: system agents' actions.
            next_timestep: data emitted by an environment during
                interaction.
            next_extras: possible extra
                information to record during the transition.
        """
        if self._adder is not None:

            # Add core states to extras
            next_extras.update(
                {"zero_padding_mask": np.array(1)}
            )

            # Adder
            self._adder.add(actions, next_timestep, next_extras)

    def select_actions(self, observations):

        actions, next_core_states = self._select_actions(observations, self._core_states)

        # Update core states
        for agent in self._core_states.keys():
            self._core_states[agent] = next_core_states[agent]

        # Convert actions to numpy
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        return actions

    def update(self, wait: bool = False) -> None:
        """Update executor variables

        Args:
            wait (bool, optional): whether to stall the executor's request for new
                variables. Defaults to False.
        """

        if self._variable_client:
            self._variable_client.update(wait)  


    # PRIVATE METHODS AND HOOKS

    @tf.function
    def _select_actions(self, observations: Dict, core_states: Dict):
        actions = {}
        next_core_states = {}
        for agent in observations.keys():
            actions[agent], next_core_states[agent] = self._select_action(
                agent, 
                observations[agent].observation,
                observations[agent].legal_actions,
                core_states[agent]
            )

        return actions, next_core_states


    def _select_action(self, agent, observation, legal_actions, core_state):

        # Add a dummy batch dimension
        observation = tf.expand_dims(observation, axis=0)
        legal_actions = tf.cast(tf.expand_dims(legal_actions, axis=0), "bool")

        # Pass observation through Q-network
        logits, next_core_state = self._policy_network(observation, core_state)

        # Mask out illegal actions
        logits = tf.where(legal_actions, logits, -1e10)
        probs = tf.nn.softmax(logits, axis=-1) # Softmax to get probs
        probs = tf.where(legal_actions, probs, 0.0) # Zero probs for illegal actions
        probs = probs / tf.reduce_sum(probs) # Renormalize 

        # Make categorical distribution        
        dist = tfp.distributions.Categorical(probs=probs, validate_args=True)
            
        # Sample action
        action = tf.cast(dist.sample(), "int64")

        return action, next_core_state