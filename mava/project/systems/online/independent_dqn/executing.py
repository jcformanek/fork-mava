from typing import Dict, Optional

import tree
import dm_env
import numpy as np
import tensorflow as tf
import sonnet as snt
from acme import types
from acme.tf import utils as tf2_utils

from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationTimestepScheduler,
)

class IndependentDQNExecutor:

    def __init__(
        self,
        agents,
        q_network: snt.Module,
        action_selectors: Dict,
        variable_client = None,
        adder = None,
        evaluator = False
    ):
        self._agents = agents
        self._q_network = q_network
        self._action_selectors = action_selectors
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
            self._core_states[agent] = self._q_network.initial_state(1)

        if self._adder is not None:
            # Convert core states to numpy arrays
            numpy_states = {
                agent: tf2_utils.to_numpy_squeeze(core_state)
                for agent, core_state in self._core_states.items()
            }

            extras.update(
                {"core_states": numpy_states, "zero_padding_mask": np.array(1)}
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
            # Convert core states to numpy arrays
            numpy_states = {
                agent: tf2_utils.to_numpy_squeeze(core_state)
                for agent, core_state in self._core_states.items()
            }

            # Add core states to extras
            next_extras.update(
                {"core_states": numpy_states, "zero_padding_mask": np.array(1)}
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

    def after_action_selection(self, time_t: int) -> None:
        """After action selection.

        Args:
            time_t: timestep
        """
        self._decrement_epsilon(time_t)

    def get_stats(self) -> Dict:
        """Return extra stats to log.

        Returns:
            epsilon information.
        """
        return {
            f"{network}_epsilon": action_selector.get_epsilon()
            for network, action_selector in self._action_selectors.items()
        }

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
        legal_actions = tf.expand_dims(legal_actions, axis=0)

        # Pass observation through Q-network
        action_values, next_core_state = self._q_network(observation, core_state)

        # Pass action values through action selector
        action = self._action_selectors[agent](action_values, legal_actions)

        return action, next_core_state

    def _decrement_epsilon(self, time_t: Optional[int]) -> None:
        """Decrements epsilon in action selectors."""
        {
            action_selector.decrement_epsilon_time_t(time_t)
            if (
                isinstance(
                    action_selector._exploration_scheduler,
                    BaseExplorationTimestepScheduler,
                )
                and time_t
            )
            else action_selector.decrement_epsilon()
            for action_selector in self._action_selectors.values()
        }