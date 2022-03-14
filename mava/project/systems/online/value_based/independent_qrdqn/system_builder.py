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
"""Independent Recurrent Quantile Regression DQN system implementation."""
from typing import Any, Callable, Dict, Optional, Type

import acme
import dm_env
import launchpad as lp
import numpy as np
import reverb
import tensorflow as tf
import sonnet as snt
from acme import specs as acme_specs
from acme import datasets
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.modules.exploration.exploration_scheduling import (
    ConstantScheduler,
)
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.types import EpsilonScheduler
from mava.utils.loggers import MavaLogger
from mava.wrappers import DetailedPerAgentStatistics, ScaledDetailedTrainerStatistics
from mava.utils.builder_utils import initialize_epsilon_schedulers
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy
from mava.project.systems.online.value_based.independent_dqn import IndependentDQN
from mava.project.systems.online.value_based.independent_qrdqn import IndependentQRDQNExecutor, IndependentQRDQNTrainer
from mava.project.components.environment_loops import EnvironmentLoop


class IndependentQRDQN(IndependentDQN):
    """Independent recurrent QR-DQN system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        num_atoms: int = 51,
        exploration_scheduler: EpsilonScheduler = LinearExplorationScheduler(
            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=1e-5,
        ),
        logger_factory: MavaLogger = None,
        discount: float = 0.99,
        batch_size: int = 32,
        min_replay_size: int = 64,
        max_replay_size: int = 5000,
        target_averaging: bool = False,
        target_update_period: int = 100,
        target_update_rate: Optional[float] = None,
        executor_variable_update_period: int = 1000,
        samples_per_insert: Optional[float] = 4.0,
        optimizer: snt.Optimizer = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        sequence_length: int = 20,
        period: int = 10,
        max_gradient_norm: float = 20,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 5,
        train_loop_fn: Callable = EnvironmentLoop,
        eval_loop_fn: Callable = EnvironmentLoop,
        termination_condition: Optional[Dict[str, int]] = None,
        evaluator_interval: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        """Initialise the system.

        NB we assume we are using a shared network for all agents in the system. If there
        are different agent types, then consider concatenating agent IDs to the observations
        so that the network can learn different behaviour per agent.

        Args:
            TODO
        """
        super().__init__(
            environment_factory=environment_factory,
            exploration_scheduler=exploration_scheduler,
            logger_factory=logger_factory,
            discount=discount,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            executor_variable_update_period=executor_variable_update_period,
            samples_per_insert=samples_per_insert,
            optimizer=optimizer,
            sequence_length=sequence_length,
            period=period,
            max_gradient_norm=max_gradient_norm,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            checkpoint_minute_interval=checkpoint_minute_interval,
            train_loop_fn=train_loop_fn,
            eval_loop_fn=eval_loop_fn,
            termination_condition=termination_condition,
            evaluator_interval=evaluator_interval,
            seed=seed,
        )

        self._trainer_fn = IndependentQRDQNTrainer
        self._executor_fn = IndependentQRDQNExecutor
        self._num_atoms = num_atoms

    # HOOKS

    def _initialise_q_network(self):

        spec = list(self._environment_spec.get_agent_specs().values())[0]
        num_actions = spec.actions.num_values
        dummy_observation = tf.expand_dims(tf.zeros_like(spec.observations.observation), axis=0)

        q_network = snt.DeepRNN(
            [
                snt.Linear(64),
                snt.GRU(64),
                snt.Linear(num_actions * self._num_atoms),
            ]
        )
        # Dummy recurent core state
        dummy_core_state = q_network.initial_state(1)

        # Initialize variables
        q_network(dummy_observation, dummy_core_state)

        return q_network

    def _extra_trainer_setup(self, trainer):
        """Setup atoms"""

        trainer.extra_setup(num_atoms=self._num_atoms)

        return trainer