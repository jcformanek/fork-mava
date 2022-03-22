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
from typing import Callable, Dict, Optional

import dm_env
import sonnet as snt

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationTimestepScheduler,
)
from mava.types import EpsilonScheduler
from mava.utils.loggers import MavaLogger
from mava.project.systems.online.independent_dqn import IndependentDQN, IndependentDQNExecutor
from mava.project.systems.online.vdn import VDNTrainer
from mava.project.components.environment_loops import EnvironmentLoop


class VDN(IndependentDQN):
    """Independent recurrent DQN system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        exploration_scheduler=LinearExplorationTimestepScheduler(
            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_steps=50_000,
        ),
        wandb: bool = False,
        logger_factory: MavaLogger = None,
        discount: float = 0.99,
        batch_size: int = 32,
        min_replay_size: int = 64,
        max_replay_size: int = 5000,
        target_averaging: bool = False,
        target_update_period: int = 200,
        target_update_rate: Optional[float] = None,
        executor_variable_update_period: int = 1000,
        samples_per_insert: Optional[float] = None,
        optimizer: snt.Optimizer = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        sequence_length: int = 20,
        period: int = 10,
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
        offline_environment_logging = False,
        offline_environment_logging_kwargs = {}
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
            wandb=wandb,
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
            lambda_=lambda_,
            termination_condition=termination_condition,
            evaluator_interval=evaluator_interval,
            seed=seed,
            offline_environment_logging=offline_environment_logging,
            offline_environment_logging_kwargs=offline_environment_logging_kwargs
        )

        self._trainer_fn=VDNTrainer
        self._executor_fn=IndependentDQNExecutor