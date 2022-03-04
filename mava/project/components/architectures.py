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

"""Decentralised architectures for multi-agent RL systems"""

import copy
from typing import Dict

import sonnet as snt
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.components.tf.architectures import (
    BaseArchitecture,
)
from mava.types import OLT


class DecentralisedConstrainedValueActor(BaseArchitecture):
    """Decentralised (independent) constrained value-based multi-agent actor architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        value_networks: Dict[str, snt.Module],
        behaviour_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
    ):
        self._env_spec = environment_spec
        self._agents = self._env_spec.get_agent_ids()
        self._agent_types = self._env_spec.get_agent_types()
        self._agent_specs = self._env_spec.get_agent_specs()
        self._agent_type_specs = self._env_spec.get_agent_type_specs()

        self._value_networks = value_networks
        self._behaviour_networks = behaviour_networks
        self._action_selectors = action_selectors
        self._observation_networks = observation_networks
        self._agent_net_keys = agent_net_keys
        self._n_agents = len(self._agents)

        self._create_target_networks()

    def _create_target_networks(self) -> None:
        # create target behaviour networks
        self._target_value_networks = copy.deepcopy(self._value_networks)
        self._target_observation_networks = copy.deepcopy(self._observation_networks)

    def _get_actor_specs(self) -> Dict[str, OLT]:
        actor_obs_specs = {}
        for agent_key in self._agents:
            # Get observation spec for policy.
            actor_obs_specs[agent_key] = self._agent_specs[
                agent_key
            ].observations.observation
        return actor_obs_specs

    def create_actor_variables(self) -> Dict[str, Dict[str, snt.Module]]:

        actor_networks: Dict[str, Dict[str, snt.Module]] = {
            "values": {},
            "behaviours": {},
            "target_values": {},
            "observations": {},
            "target_observations": {},
            "selectors": {},
        }

        # get actor specs
        actor_obs_specs = self._get_actor_specs()

        # create policy variables for each agent
        for agent_key in self._agents:
            agent_net_key = self._agent_net_keys[agent_key]
            obs_spec = actor_obs_specs[agent_key]
            # Create variables for observation and value networks.
            embed = tf2_utils.create_variables(
                self._observation_networks[agent_net_key], [obs_spec]
            )
            tf2_utils.create_variables(self._value_networks[agent_net_key], [embed])

            # Create bahaviour network variables
            tf2_utils.create_variables(self._behaviour_networks[agent_net_key], [embed])

            # Create target value and observation network variables
            embed = tf2_utils.create_variables(
                self._target_observation_networks[agent_net_key], [obs_spec]
            )
            tf2_utils.create_variables(
                self._target_value_networks[agent_net_key], [embed]
            )

        actor_networks["values"] = self._value_networks
        actor_networks["behaviours"] = self._behaviour_networks
        actor_networks["target_values"] = self._target_value_networks
        actor_networks["selectors"] = self._action_selectors
        actor_networks["observations"] = self._observation_networks
        actor_networks["target_observations"] = self._target_observation_networks

        return actor_networks

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        networks = self.create_actor_variables()
        return networks