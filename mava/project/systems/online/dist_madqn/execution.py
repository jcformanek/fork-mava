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
from mava.systems.tf.madqn import MADQNRecurrentExecutor
from mava.systems.tf import executors
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray
tfd = tfp.distributions

class DistMADQNExecutor(MADQNRecurrentExecutor):
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
        network_sampling_setup: List,
        net_keys_to_ids: Dict[str, int],
        evaluator: bool = False,
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
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
        super().__init__(
            observation_networks=observation_networks,
            action_selectors=action_selectors,
            value_networks=value_networks,
            agent_specs=agent_specs,
            agent_net_keys=agent_net_keys,
            network_sampling_setup=network_sampling_setup,
            net_keys_to_ids=net_keys_to_ids,
            evaluator=evaluator,
            adder=adder,
            counts=counts,
            variable_client=variable_client,
            store_recurrent_state=store_recurrent_state,
            interval=interval,
        )

        self._atoms = None

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
        logits, new_state = self._value_networks[agent_key](embed, state)
        logits = tf.reshape(logits, shape=(*logits.shape[:-1], -1, len(self._atoms))) # TODO don't hardcode
        action_values_probs = tf.nn.softmax(logits)
        action_values_mean = tf.reduce_sum(action_values_probs * self._atoms, -1)


        # Pass action values through action selector
        action = self._action_selectors[agent](action_values_mean, batched_legal_actions)

        return action, new_state

class QRDQNExecutor(DistMADQNExecutor):
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
        network_sampling_setup: List,
        net_keys_to_ids: Dict[str, int],
        evaluator: bool = False,
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
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
        super().__init__(
            observation_networks=observation_networks,
            action_selectors=action_selectors,
            value_networks=value_networks,
            agent_specs=agent_specs,
            agent_net_keys=agent_net_keys,
            network_sampling_setup=network_sampling_setup,
            net_keys_to_ids=net_keys_to_ids,
            evaluator=evaluator,
            adder=adder,
            counts=counts,
            variable_client=variable_client,
            store_recurrent_state=store_recurrent_state,
            interval=interval,
        )

        self._atoms = None

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
        q, new_state = self._value_networks[agent_key](embed, state)
        q = tf.reshape(q, shape=(*q.shape[:-1], -1, self._atoms)) # TODO don't hardcode
        action_values = tf.reduce_mean(q, -1)

        # Pass action values through action selector
        action = self._action_selectors[agent](action_values, batched_legal_actions)

        return action, new_state