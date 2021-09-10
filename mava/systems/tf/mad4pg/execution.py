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

"""MAD4PG system executor implementation."""

from typing import Any, Dict, List, Optional

import sonnet as snt
from acme.specs import EnvironmentSpec
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.systems.tf.maddpg.execution import (
    MADDPGFeedForwardExecutor,
    MADDPGRecurrentExecutor,
)


class MAD4PGFeedForwardExecutor(MADDPGFeedForwardExecutor):
    """A feed-forward executor for MAD4PG.
    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        net_to_ints: Dict[str, int],
        executor_samples: List,
        adder: Optional[adders.ParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):

        """Initialise the system executor
        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_specs (Dict[str, EnvironmentSpec]): agent observation and action
                space specifications.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
        """

        super().__init__(
            policy_networks=policy_networks,
            agent_specs=agent_specs,
            adder=adder,
            variable_client=variable_client,
            counts=counts,
            agent_net_keys=agent_net_keys,
            executor_samples=executor_samples,
            net_to_ints=net_to_ints,
        )


class MAD4PGRecurrentExecutor(MADDPGRecurrentExecutor):
    """A recurrent executor for MAD4PG.
    An executor based on a recurrent policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        executor_samples: List,
        net_to_ints: Dict[str, int],
        adder: Optional[adders.ParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        environment=None,
        executor_id=None,
    ):
        """Initialise the system executor

        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_specs (Dict[str, EnvironmentSpec]): agent observation and action
                space specifications.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
        """

        super().__init__(
            policy_networks=policy_networks,
            agent_specs=agent_specs,
            adder=adder,
            variable_client=variable_client,
            counts=counts,
            agent_net_keys=agent_net_keys,
            executor_samples=executor_samples,
            net_to_ints=net_to_ints,
            environment=environment,
            executor_id=executor_id,
        )
