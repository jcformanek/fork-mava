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

"""Tests for MAPPO."""

import functools
from typing import Dict, Sequence, Union

import dm_env
import launchpad as lp
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme.tf import networks
from acme.tf import utils as tf2_utils
from launchpad.nodes.python.local_multi_processing import PythonProcess

import mava
from mava import specs as mava_specs
from mava.systems.tf import mappo
from mava.utils import lp_utils
from mava.utils.environments import debugging_utils


class TestMAPPO:
    """Simple integration/smoke test for MAPPO."""

    def test_mappo_on_debugging_env(self) -> None:
        """Tests that the system can run on the simple spread
        debugging environment without crashing."""

        # environment
        environment_factory = functools.partial(
            debugging_utils.make_environment,
            env_name="simple_spread",
            action_space="discrete",
        )

        # networks
        network_factory = lp_utils.partial_kwargs(make_networks)

        # system
        system = mappo.MAPPO(
            environment_factory=environment_factory,
            network_factory=network_factory,
            num_executors=2,
            batch_size=32,
            max_queue_size=1000,
            policy_optimizer=snt.optimizers.Adam(learning_rate=1e-3),
            critic_optimizer=snt.optimizers.Adam(learning_rate=1e-3),
            checkpoint=False,
        )
        program = system.build()

        (trainer_node,) = program.groups["trainer"]
        trainer_node.disable_run()

        # Launch gpu config - don't use gpu
        gpu_id = -1
        env_vars = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
        local_resources = {
            "trainer": PythonProcess(env=env_vars),
            "evaluator": PythonProcess(env=env_vars),
            "executor": PythonProcess(env=env_vars),
        }

        lp.launch(
            program,
            launch_type="test_mt",
            local_resources=local_resources,
        )

        trainer: mava.Trainer = trainer_node.create_handle().dereference()

        for _ in range(5):
            trainer.step()
