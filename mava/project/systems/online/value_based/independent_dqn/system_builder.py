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
    LinearExplorationScheduler,
)
from mava.types import EpsilonScheduler
from mava.utils.loggers import MavaLogger
from mava.wrappers import DetailedPerAgentStatistics
from mava.utils.builder_utils import initialize_epsilon_schedulers
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy
from mava.project.systems.online.value_based.independent_dqn import IndependentDQNTrainer, IndependentDQNExecutor
from mava.project.components.environment_loops import EnvironmentLoop


class IndependentDQN:
    """Independent recurrent DQN system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        exploration_scheduler: EpsilonScheduler = LinearExplorationScheduler(
            epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=1e-5,
        ),
        logger_factory: MavaLogger = None,
        discount: float = 0.99,
        batch_size: int = 32,
        min_replay_size: int = 64,
        max_replay_size: int = 5000,
        target_averaging: bool = False,
        target_update_period: int = 200,
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

        # Environment and agents
        self._environment_spec = mava_specs.MAEnvironmentSpec(
            environment_factory(evaluation=False)  # type: ignore
        )
        self._agents = self._environment_spec.get_agent_ids()
        self._agent_net_keys = {agent: "shared_network" for agent in self._agents}

        # Factories
        self._environment_factory = environment_factory
        self._logger_factory = logger_factory

        # Config
        self._discount = discount
        self._batch_size = batch_size
        self._target_averaging = target_averaging
        self._target_update_period = target_update_period
        self._target_update_rate = target_update_rate
        self._exploration_scheduler = exploration_scheduler
        self._executor_variable_update_period = executor_variable_update_period
        self._min_replay_size = min_replay_size
        self._max_replay_size = max_replay_size
        self._samples_per_insert = samples_per_insert
        self._sequence_length = sequence_length
        self._period = period
        self._max_gradient_norm = max_gradient_norm
        self._optimizer = optimizer

        # Checkpointing
        self._checkpoint = checkpoint
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint_minute_interval = checkpoint_minute_interval
        # TODO

        # Trainer and Executor fns
        self._trainer_fn = IndependentDQNTrainer
        self._executor_fn = IndependentDQNExecutor
        self._train_loop_fn = train_loop_fn
        self._eval_loop_fn = eval_loop_fn
        self._evaluator_interval = evaluator_interval # TODO 
        self._termination_condition = termination_condition # TODO

        # Random seed
        self._seed = seed

    def replay(self) -> Any:
        """Create reverb replay table.

        Returns:
            Reverb replay table
        """
        adder_signiture = reverb_adders.ParallelSequenceAdder.signature(
            self._environment_spec, self._sequence_length, self._get_extras_spec()
        )

        if self._samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            rate_limiter = reverb.rate_limiters.MinSize(self._min_replay_size)
        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._samples_per_insert
            error_buffer = self._min_replay_size * samples_per_insert_tolerance

            rate_limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._min_replay_size,
                samples_per_insert=self._samples_per_insert,
                error_buffer=error_buffer,
            )

        replay_table = reverb.Table(
            name='priority_table',
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._max_replay_size,
            rate_limiter=rate_limiter,
            signature=adder_signiture,
        )

        return [replay_table]

    def executor(
        self,
        replay_client: reverb.Client,
        variable_source: acme.VariableSource,
    ) -> mava.ParallelEnvironmentLoop:
        """System executor.

        Args:
            executor_id: id to identify the executor process for logging purposes.
            replay: replay data table to push data to.
            variable_source: variable server for updating
                network variables.

        Returns:
            mava.ParallelEnvironmentLoop: environment-executor loop instance.
        """
        # Create the environment.
        environment = self._environment_factory(evaluation=False)  # type: ignore

        # ENV LOGGING
        # environment = MAEnvironmentLogger(environment, max_trajectory_length=120, trajectories_per_file=1000) 

        # Create logger
        logger = self._logger_factory("executor")

        # Create replay adder
        adder = self._build_adder(replay_client=replay_client) # hook

        # Create the executor.
        executor = self._build_executor(
            variable_source, 
            adder=adder, 
            exploration_scheduler=self._exploration_scheduler,
            variable_update_period=self._executor_variable_update_period
        ) # hook

        # Create the loop to connect environment and executor
        executor_environment_loop = self._train_loop_fn(
            environment,
            executor,
            logger=logger,
        )

        # Environment loop statistics
        executor_environment_loop = DetailedPerAgentStatistics(executor_environment_loop)

        return executor_environment_loop

    def evaluator(
        self,
        variable_source: acme.VariableSource,
    ) -> Any:
        """System evaluator.

        Args:
            variable_source: variable server for updating
                network variables.
            logger: logger object.

        Returns:
            environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """
        # Executor with no exploration and no adder
        executor = self._build_executor(variable_source, evaluator=True) # hook

        # Make the environment
        environment = self._environment_factory()  # type: ignore

        # Create logger and counter
        logger = self._logger_factory("evaluator")

        # Create the loop to connect environment and executor
        executor_environment_loop = self._eval_loop_fn(
            environment,
            executor,
            logger=logger,
        )

        # Environment loop statistics
        executor_environment_loop = DetailedPerAgentStatistics(executor_environment_loop)

        return executor_environment_loop

    def trainer(
        self,
        replay_client: reverb.Client,
    ) -> mava.core.Trainer:
        """System trainer.

        Args:
            replay: replay data table to pull data from.

        Returns:
            system trainer.
        """
        # Create logger
        logger = self._logger_factory("trainer")

        # Build dataset
        dataset = datasets.make_reverb_dataset(
            table="priority_table",
            server_address=replay_client.server_address,
            batch_size=self._batch_size,
            prefetch_size=4,
            sequence_length=self._sequence_length,
        )

        # Make the trainer
        trainer = self._build_trainer(dataset, logger) # hook

        return trainer

    def build(self, name: str = "madqn") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name: system name.

        Returns:
            graph program for distributed system training.
        """
        program = lp.Program(name=name)

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group("trainer"):
            # Add trainer
            trainer = program.add_node(
                lp.CourierNode(self.trainer, replay_client=replay)
            )

        with program.group("evaluator"):
            # Add evaluator
            program.add_node(lp.CourierNode(self.evaluator, variable_source=trainer))

        with program.group("executor"):
            # Add executor
            program.add_node(
                lp.CourierNode(self.executor, replay_client=replay, variable_source=trainer)
            )

        return program

    def run_single_proc_system(self, training_steps_per_episode = 1, evaluator_period=5):
        
        replay_tables = self.replay()
        replay_server = reverb.Server(tables=replay_tables)
        replay_client = reverb.Client(f'localhost:{replay_server.port}')

        trainer = self.trainer(replay_client)

        executor = self.executor(replay_client, trainer)

        evaluator = self.evaluator(trainer)

        episode = 0
        while True:

            episode += 1

            episode_stats = executor.run_episode()
            executor._logger.write(episode_stats)

            if episode >= self._min_replay_size:
                for _ in range(training_steps_per_episode):
                    trainer.step()

            if episode % evaluator_period == 0:
                episode_stats = evaluator.run_episode()
                evaluator._logger.write(episode_stats)

    # PRIVATE METHODS AND HOOKS

    def _initialise_q_network(self):

        spec = list(self._environment_spec.get_agent_specs().values())[0]
        num_actions = spec.actions.num_values
        dummy_observation = tf.expand_dims(tf.zeros_like(spec.observations.observation), axis=0)

        q_network = snt.DeepRNN(
            [
                snt.Linear(64),
                snt.GRU(64),
                snt.Linear(num_actions)
            ]
        )
        # Dummy recurent core state
        dummy_core_state = q_network.initial_state(1)

        # Initialize variables
        q_network(dummy_observation, dummy_core_state)

        return q_network
    
    def _get_extras_spec(self) -> Any:
        """Helper to establish specs for extras.

        Returns:
            Dictionary containing extras spec
        """
        q_network = self._initialise_q_network() # hook

        core_state_specs = {}
        for agent in self._agents:
            core_state_specs[agent] = (
                tf2_utils.squeeze_batch_dim(
                    q_network.initial_state(1)
                ),
            )
        return {"core_states": core_state_specs, "zero_padding_mask": np.array(1)}


    def _build_adder(self, replay_client):

        adder = reverb_adders.ParallelSequenceAdder(
            priority_fns=None,
            client=replay_client,
            sequence_length=self._sequence_length,
            period=self._period,
        )

        return adder

    def _build_variable_client(self, variable_source, network, update_period=1):
        # Get variables
        variables = {"q_network": network.variables}

        # Make variable client
        variable_client = variable_utils.VariableClient(
            client=variable_source,
            variables=variables,
            update_period=update_period,
        )

        # Make sure not to use a random policy after checkpoint restoration by
        # assigning variables before running the environment loop.
        variable_client.update_and_wait()

        return variable_client

    def _build_executor(self, variable_source, evaluator=False, adder=None, exploration_scheduler=None, variable_update_period=1):

        # Exploration scheduler
        if exploration_scheduler is None:
            # Constant zero exploration
            exploration_schedules = {agent: ConstantScheduler(epsilon=0.0) for agent in self._agents}
        else:
            exploration_schedules = {agent: exploration_scheduler for agent in self._agents}

        # Epsilon-greedy action selector
        action_selector = {"shared_network": EpsilonGreedy}

        # Action selectors with epsilon schedulers (one per agent)
        action_selectors_with_epsilon_schedulers = initialize_epsilon_schedulers(
            exploration_schedules,
            action_selector,
            agent_net_keys=self._agent_net_keys,
            seed=self._seed,
        )

        # Initialise Q-network
        q_network = self._initialise_q_network() # hook

        # Variable client
        variable_client = self._build_variable_client(variable_source, q_network, update_period=variable_update_period) # hook

        # Executor
        executor =  self._executor_fn(
            agents=self._agents,
            q_network=q_network,
            action_selectors=action_selectors_with_epsilon_schedulers,
            variable_client=variable_client,
            adder=adder,
            evaluator=evaluator
        )

        return executor

    def _extra_trainer_setup(self, trainer):
        """NoOp"""
        return trainer

    def _build_trainer(self, dataset, logger):
        # Create the network
        q_network = self._initialise_q_network() # hook

        trainer =  self._trainer_fn(
                agents=self._agents,
                q_network=q_network,
                optimizer=self._optimizer,
                discount=self._discount,
                target_averaging=self._target_averaging,
                target_update_period=self._target_update_period,
                target_update_rate=self._target_update_rate,
                dataset=dataset,
                max_gradient_norm=self._max_gradient_norm,
                logger=logger,
            )

        trainer = self._extra_trainer_setup(trainer) # hook

        return trainer