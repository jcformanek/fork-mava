# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MADQN system executor implementation."""
from typing import Any, Dict, List, Optional, Tuple, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
from acme import types
from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from dm_env import specs

from mava import adders
from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationTimestepScheduler,
)
from mava.systems.tf import executors
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num

class DQNExecutor:
    """DQN executor."""

    def __init__(self, action_selectors: Dict) -> None:
        """Initialise DQN executor."""
        self._action_selectors = action_selectors

    def _get_epsilon(self) -> Union[float, np.ndarray]:
        """Return epsilon.

        Returns:
            epsilon values.
        """
        data = list(
            {
                action_selector.get_epsilon()
                for action_selector in self._action_selectors.values()
            }
        )
        if len(data) == 1:
            return data[0]
        else:
            return np.array(list(data))

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

class OfflineRecurrentMADQNExecutor(DQNExecutor):

    """A recurrent executor for MADQN like systems.

    An executor based on a recurrent epsilon-greedy policy
    for each agent in the system.
    """

    def __init__(
        self,
        observation_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        value_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        interval: Optional[dict] = None,
    ):
        """Initialise the system executor.

        Args:
            action_selectors: epsilon greedy action selection
            value_networks: agents value networks.
            variable_client: client for managing
                network variable distribution
            observation_networks: observation networks for each agent in
                the system.
            agent_specs: agent observation and action
                space specifications.
            agent_net_keys: specifies what network each agent uses.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
            net_keys_to_ids: Specifies a mapping from network keys to their integer id.
            adder: adder which sends data
                to a replay buffer. Defaults to None.
            counts: Count values used to record excutor episode and steps.
            variable_client:
                client to copy weights from the trainer. Defaults to None.
            store_recurrent_state: boolean to store the recurrent
                network hidden state. Defaults to True.
            evaluator: whether the executor will be used for
                evaluation.
            interval: interval that evaluations are run at.
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._counts = counts
        self._network_int_keys_extras: Dict[str, np.ndarray] = {}
        self._interval = interval
        self._value_networks = value_networks
        self._agent_net_keys = agent_net_keys
        self._variable_client = variable_client
        self._observation_networks = observation_networks
        self._action_selectors = action_selectors
        self._states: Dict[str, Any] = {}
        self._evaluator = True

    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        legal_actions: types.NestedTensor,
        state: types.NestedTensor,
    ) -> Tuple:
        """Agent epsilon-greedy policy.

        Args:
            agent: agent id
            observation: observation tensor received from the
                environment.
            legal_actions: one-hot vector of legal actions
            state: recurrent network state.

        Returns:
            action, policy and new recurrent hidden state
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legal_actions = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

        # Pass through observation network
        embed = self._observation_networks[agent_key](batched_observation)

        # Compute the policy, conditioned on the observation.
        action_values, new_state = self._value_networks[agent_key](embed, state)

        # Pass action values through action selector
        action = self._action_selectors[agent](action_values, batched_legal_actions)

        return action, new_state

    @tf.function
    def _select_actions(
        self,
        observations: Dict[str, types.NestedArray],
        states: Dict[str, types.NestedArray],
    ) -> types.NestedArray:
        """The part of select_action that we can do inside tf.function"""
        actions: Dict = {}
        new_states: Dict = {}
        for agent, observation in observations.items():
            actions[agent], new_states[agent] = self._policy(
                agent,
                observation.observation,
                observation.legal_actions,
                states[agent],
            )
        return actions, new_states

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> types.NestedArray:
        """Select the actions for all agents in the system

        Args:
            observations: agent observations from the
                environment.

        Returns:
            actions and policies for all agents in the system.
        """

        actions, new_states = self._select_actions(observations, self._states)

        # Convert actions to numpy arrays
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)

        # Update agent core state
        for agent, state in new_states.items():
            self._states[agent] = state

        return actions

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record first observed timestep from the environment

        Args:
            timestep: data emitted by an environment at first step of
                interaction.
            extras: possible extra information
                to record during the first step.
        """
        # Re-initialize the RNN state.
        for agent, _ in timestep.observation.items():
            # index network either on agent type or on agent id
            agent_key = self._agent_net_keys[agent]
            self._states[agent] = self._value_networks[agent_key].initial_state(1)

        return

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
        return

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            self._variable_client.get_async()