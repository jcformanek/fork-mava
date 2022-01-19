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

"""Commonly used adder components for system builders"""
import abc

from typing import Dict, Optional, Any

from mava import adders
from mava.callbacks import Callback
from mava.core import SystemExecutor


class Executor(Callback):
    def __init__(
        self,
        policy_networks: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[Any] = None,
    ) -> None:

        # Store these for later use.
        self._policy_networks = policy_networks
        self._agent_net_keys = agent_net_keys
        self._adder = adder
        self._variable_client = variable_client

    @abc.abstractmethod
    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        """[summary]

        Args:
            executor (SystemExecutor): [description]
        """