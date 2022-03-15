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
from mava.project.components.epsilon_greedy import EpsilonGreedy
from mava.project.systems.online.actor_critic.independent_offpg import IndependentOffPGTrainer, IndependentOffPGExecutor
from mava.project.systems.online.value_based.independent_dqn import IndependentDQN
from mava.project.components.environment_loops import EnvironmentLoop


class IndependentOffPG(IndependentDQN):
    """Independent recurrent Off-policy gradients."""

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
        q_optimizer: snt.Optimizer = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        policy_optimizer: snt.Optimizer = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        sequence_length: int = 61,
        period: int = 61,
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
            optimizer=q_optimizer, # super stores the Q-optimizer
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

        self._trainer_fn = IndependentOffPGTrainer
        self._executor_fn = IndependentOffPGExecutor

        # Policy optimizer
        self._policy_optimizer = policy_optimizer
    
    # PRIVATE METHODS AND HOOKS
    def _get_extras_spec(self) -> Any:
        """Helper to establish specs for extras.

        Returns:
            Dictionary containing extras spec
        """
        policy_network = self._initialise_policy_network() # hook

        core_state_specs = {}
        for agent in self._agents:
            core_state_specs[agent] = (
                tf2_utils.squeeze_batch_dim(
                    policy_network.initial_state(1)
                ),
            )

        return {"core_states": core_state_specs, "zero_padding_mask": np.array(1)}

    def _initialise_q_network(self):

        spec = list(self._environment_spec.get_agent_specs().values())[0]
        num_actions = spec.actions.num_values
        dummy_observation = tf.zeros_like(spec.observations.observation)

        state_spec = self._environment_spec.get_extra_specs()["s_t"]
        dummy_state = tf.zeros_like(state_spec)

        # Concat state to observation
        dummy_input = tf.concat([dummy_state, dummy_observation], axis=-1)

        # Add batch dim
        dummy_input = tf.expand_dims(dummy_input, axis=0)

        q_network = snt.Sequential(
            [
                snt.Linear(256),
                tf.keras.layers.ReLU(),
                snt.Linear(256),
                tf.keras.layers.ReLU(),
                snt.Linear(num_actions)
            ]
        )

        # Create variables
        q_network(dummy_input)

        return q_network

    def _initialise_policy_network(self):
        spec = list(self._environment_spec.get_agent_specs().values())[0]
        num_actions = spec.actions.num_values
        dummy_observation = tf.expand_dims(tf.zeros_like(spec.observations.observation), axis=0)

        policy_network = snt.DeepRNN(
            [
                snt.Linear(64),
                snt.GRU(64),
                snt.Linear(num_actions)
            ]
        )
        # Dummy recurent core state
        dummy_core_state = policy_network.initial_state(1)

        # Initialize variables
        policy_network(dummy_observation, dummy_core_state)

        return policy_network

    def _build_executor(self, variable_source, evaluator=False, adder=None, exploration_scheduler=None, variable_update_period=1):
        # TODO make this function more general so that we dont need to overwrite it from super

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

        # Add policy network to variable client
        policy_network = self._initialise_policy_network()
        # Variable client
        variable_client = self._build_variable_client(variable_source, policy_network, update_period=variable_update_period) # hook

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

    def _build_variable_client(self, variable_source, network, update_period):
        # Get variables
        variables = {"policy_network": network.variables}

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

    def _extra_executor_setup(self, executor, evaluator=False):
        # Initialise policy network
        policy_network = self._initialise_policy_network()

        executor.extra_setup(policy_network=policy_network)

        return executor

    def _extra_trainer_setup(self, trainer):
        policy_network = self._initialise_policy_network() # hook

        trainer.extra_setup(
            policy_network=policy_network, 
            policy_optimizer=self._policy_optimizer,
            lambda_=0.8
        )
        return trainer