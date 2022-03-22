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
"""Independent Recurrent DQN system implementation."""
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
    LinearExplorationTimestepScheduler,
)
from mava.types import EpsilonScheduler
from mava.utils.loggers import MavaLogger
from mava.wrappers import DetailedPerAgentStatistics
from mava.utils.builder_utils import initialize_epsilon_schedulers
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy
from mava.project.systems.online.independent_dqn import IndependentDQN
from mava.project.components.environment_loops import EnvironmentLoop
from mava.project.components.offline import MAOfflineEnvironmentDataset
from mava.project.systems.offline.offline_system import OfflineSystem
from mava.project.systems.offline.independent_dqn.training import BatchConstrainedIndependentDQNTrainer

import wandb


class OfflineIndependentDQN(OfflineSystem, IndependentDQN):
    """Independent recurrent DQN system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        offline_env_log_dir: str,
        shuffle_buffer_size: int = 10_000,
        wandb: bool = False,
        logger_factory: MavaLogger = None,
        discount: float = 0.99,
        batch_size: int = 32,
        target_averaging: bool = False,
        target_update_period: int = 200,
        target_update_rate: Optional[float] = None,
        optimizer: snt.Optimizer = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        max_gradient_norm: float = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 5,
        train_loop_fn: Callable = EnvironmentLoop,
        eval_loop_fn: Callable = EnvironmentLoop,
        lambda_: Optional[float] = None,
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
            logger_factory=logger_factory,
            wandb=wandb,
            discount=discount,
            batch_size=batch_size,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            optimizer=optimizer,
            max_gradient_norm=max_gradient_norm,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            checkpoint_minute_interval=checkpoint_minute_interval,
            train_loop_fn=train_loop_fn,
            eval_loop_fn=eval_loop_fn,
            termination_condition=termination_condition,
            evaluator_interval=evaluator_interval,
            lambda_=lambda_,
            seed=seed,
        )

        self._offline_env__log_dir = offline_env_log_dir
        self._shuffle_buffer_size = shuffle_buffer_size

        self._trainer_fn = BatchConstrainedIndependentDQNTrainer

    def _extra_trainer_setup(self, trainer):
        behaviour_cloning_network = self._initialise_behaviour_cloning_network()
        behaviour_cloning_optimizer = snt.optimizers.Adam(1e-4)
        trainer.extra_setup(threshold=0.3, lambda_=self._lambda, behaviour_cloning_network=behaviour_cloning_network, behaviour_cloning_optimizer=behaviour_cloning_optimizer)

        return trainer

    def _initialise_behaviour_cloning_network(self):
        spec = list(self._environment_spec.get_agent_specs().values())[0]
        num_actions = spec.actions.num_values
        dummy_observation = tf.expand_dims(tf.zeros_like(spec.observations.observation), axis=0)

        behaviour_cloning_network = snt.DeepRNN(
            [
                snt.Linear(64),
                snt.GRU(64),
                snt.Linear(num_actions),
                tf.keras.layers.Softmax()
            ]
        )
        # Dummy recurent core state
        dummy_core_state = behaviour_cloning_network.initial_state(1)

        # Initialize variables
        behaviour_cloning_network(dummy_observation, dummy_core_state)

        return behaviour_cloning_network