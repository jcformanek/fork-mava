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

"""MAD4PG system implementation."""

from typing import Callable, Dict, List, Optional, Type, Union

import dm_env
import sonnet as snt
from acme import specs as acme_specs

from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedQValueActorCritic
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf.mad4pg import training
from mava.systems.tf.maddpg.execution import MADDPGFeedForwardExecutor
from mava.systems.tf.maddpg.system import MADDPG
from mava.utils import enums
from mava.utils.loggers import MavaLogger


class MAD4PG(MADDPG):
    """MAD4PG system."""

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[
            DecentralisedQValueActorCritic
        ] = DecentralisedQValueActorCritic,
        trainer_fn: Union[
            Type[training.MAD4PGBaseTrainer],
            Type[training.MAD4PGBaseRecurrentTrainer],
        ] = training.MAD4PGDecentralisedTrainer,
        executor_fn: Type[core.Executor] = MADDPGFeedForwardExecutor,
        num_executors: int = 1,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        trainer_networks: Union[
            Dict[str, List], enums.Trainer
        ] = enums.Trainer.single_trainer,
        network_sampling_setup: Union[
            List, enums.NetworkSampler
        ] = enums.NetworkSampler.fixed_agent_networks,
        shared_weights: bool = True,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_averaging: bool = False,
        target_update_period: int = 100,
        target_update_rate: Optional[float] = None,
        executor_variable_update_period: int = 1000,
        min_replay_size: int = 1000,
        max_replay_size: int = 100000,
        samples_per_insert: Optional[float] = 32.0,
        policy_optimizer: Union[
            snt.Optimizer, Dict[str, snt.Optimizer]
        ] = snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer: Union[
            snt.Optimizer, Dict[str, snt.Optimizer]
        ] = snt.optimizers.Adam(learning_rate=1e-4),
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 20,
        bootstrap_n: int = 10,
        sigma: float = 0.3,
        max_gradient_norm: float = None,
        checkpoint: bool = True,
        checkpoint_minute_interval: int = 5,
        checkpoint_subpath: str = "~/mava/",
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
    ):
        """Initialise the system

        Args:
            environment_factory: function to
                instantiate an environment.
            network_factory: function to instantiate system networks.
            logger_factory: function to
                instantiate a system logger.
            architecture:
                system architecture, e.g. decentralised or centralised.
            trainer_fn: training type
                associated with executor and architecture, e.g. centralised training.
            executor_fn: executor type, e.g.
                feedforward or recurrent.
            num_executors: number of executor processes to run in
                parallel..
            environment_spec: description of
                the action, observation spaces etc. for each agent in the system.
            trainer_networks: networks each trainer trains on.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
            shared_weights: whether agents should share weights or not.
                When network_sampling_setup are provided the value of shared_weights is
                ignored.
            discount: discount factor to use for TD updates.
            batch_size: sample batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_averaging: whether to use polyak averaging for
                target network updates.
            target_update_period: number of steps before target
                networks are updated.
            target_update_rate: update rate when using
                averaging.
            executor_variable_update_period: number of steps before
                updating executor variables from the variable source.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take
                from replay for every insert that is made.
            policy_optimizer: optimizer(s) for updating policy networks.
            critic_optimizer: optimizer for updating critic
                networks.
            n_step: number of steps to include prior to boostrapping.
            sequence_length: recurrent sequence rollout length.
            period: Consecutive starting points for overlapping
                rollouts across a sequence.
            bootstrap_n: Used to determine the spacing between
                q_value/value estimation for bootstrapping. Should be less
                than sequence_length.
            sigma: Gaussian sigma parameter.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            checkpoint: whether to checkpoint models.
            checkpoint_minute_interval: The number of minutes to wait between
                checkpoints.
            checkpoint_subpath: subdirectory specifying where to store
                checkpoints.
            logger_config: additional configuration settings for the
                logger factory.
            train_loop_fn: function to instantiate a train loop.
            eval_loop_fn: function to instantiate an evaluation
                loop.
            train_loop_fn_kwargs: possible keyword arguments to send
                to the training loop.
            eval_loop_fn_kwargs: possible keyword arguments to send to
            the evaluation loop.
        """

        super().__init__(
            environment_factory=environment_factory,
            network_factory=network_factory,
            logger_factory=logger_factory,
            architecture=architecture,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            num_executors=num_executors,
            environment_spec=environment_spec,
            trainer_networks=trainer_networks,
            network_sampling_setup=network_sampling_setup,
            shared_weights=shared_weights,
            discount=discount,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            target_update_period=target_update_period,
            executor_variable_update_period=executor_variable_update_period,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            samples_per_insert=samples_per_insert,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            n_step=n_step,
            sequence_length=sequence_length,
            bootstrap_n=bootstrap_n,
            period=period,
            sigma=sigma,
            max_gradient_norm=max_gradient_norm,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            checkpoint_minute_interval=checkpoint_minute_interval,
            logger_config=logger_config,
            train_loop_fn=train_loop_fn,
            eval_loop_fn=eval_loop_fn,
            train_loop_fn_kwargs=train_loop_fn_kwargs,
            eval_loop_fn_kwargs=eval_loop_fn_kwargs,
            target_averaging=target_averaging,
            target_update_rate=target_update_rate,
        )
