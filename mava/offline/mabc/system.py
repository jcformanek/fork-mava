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

"""MA Batch Constrained Q-learning system implementation."""

import functools
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import tensorflow as tf
import acme
import dm_env
import launchpad as lp
import numpy as np
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils
from dm_env import specs

import mava
from mava import core
from mava.offline.offline_utils import MAEnvironmentLoggerDataset
from mava import specs as mava_specs
from mava.systems.tf import variable_utils
from mava.utils.builder_utils import initialize_epsilon_schedulers
from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.exploration.exploration_scheduling import (
    ConstantScheduler,
)
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.madqn import builder, training
from mava.systems.tf.madqn.execution import (
    MADQNRecurrentExecutor,
    sample_new_agent_keys,
)
from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.systems.tf.variable_sources import VariableSource as MavaVariableSource
from mava.types import EpsilonScheduler
from mava.utils import enums
from mava.utils.loggers import MavaLogger, logger_utils
from mava.utils.sort_utils import sort_str_num
from mava.wrappers import DetailedPerAgentStatistics, ScaledDetailedTrainerStatistics
from mava.offline.mabc import MABCTrainer, MABCExecutor
from mava.offline.components.architectures import DecentralisedConstrainedValueActor

class MABC:
    """MA Batch Constrained Q-learning system."""

    def __init__(  # noqa
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logdir: str,
        logger_factory: Callable[[str], MavaLogger] = None,
        trainer_fn = MABCTrainer,
        executor_fn = MABCExecutor,
        shuffle_buffer_size: int = 500,
        shared_weights: bool = True,
        batch_size: int = 32,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]] = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        max_gradient_norm: float = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 5,
        logger_config: Dict = {},
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn_kwargs: Dict = {},
        termination_condition: Optional[Dict[str, int]] = None,
        evaluator_interval: Optional[dict] = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):

        environment_spec = mava_specs.MAEnvironmentSpec(
            environment_factory(evaluation=False)  # type: ignore
        )

        # Set default logger if no logger provided
        if not logger_factory:
            logger_factory = functools.partial(
                logger_utils.make_logger,
                directory="~/mava",
                to_terminal=True,
                time_delta=10,
            )

        # Agent net keys
        agents = environment_spec.get_agent_ids()
        agent_net_keys = {
            agent: agent.split("_")[0] if shared_weights else agent
            for agent in agents
        }

        self._architecture = DecentralisedConstrainedValueActor
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._environment_spec = environment_spec
        self._shuffle_buffer_size = shuffle_buffer_size
        self._logdir = logdir
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn
        self._agents = agents
        self._agent_net_keys = agent_net_keys
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint
        self._checkpoint_minute_interval = checkpoint_minute_interval
        self._logger_config = logger_config
        self._eval_loop_fn = eval_loop_fn
        self._eval_loop_fn_kwargs = eval_loop_fn_kwargs
        self._evaluator_interval = evaluator_interval
        self._shared_weightsl = shared_weights
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._max_gradient_norm = max_gradient_norm
        self._termination_condition = termination_condition
        self._learning_rate_scheduler_fn = learning_rate_scheduler_fn

    def create_counter_variables(
        self, variables: Dict[str, tf.Variable]
    ) -> Dict[str, tf.Variable]:
        """Create counter variables.

        Args:
            variables: dictionary with variable_source
            variables in.

        Returns:
            variables: dictionary with variable_source
            variables in.
        """
        variables["trainer_steps"] = tf.Variable(0, dtype=tf.int32)
        variables["trainer_walltime"] = tf.Variable(0, dtype=tf.float32)
        variables["evaluator_steps"] = tf.Variable(0, dtype=tf.int32)
        variables["evaluator_episodes"] = tf.Variable(0, dtype=tf.int32)
        return variables

    def create_system(
        self,
    ) -> Dict:
        """Initialise the system variables from the network factory."""
        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # architecture args
        architecture_config = {
            "environment_spec": self._environment_spec,
            "observation_networks": networks["observations"],
            "value_networks": networks["values"],
            "behaviour_networks": networks["behaviours"],
            "action_selectors": networks["action_selectors"],
            "agent_net_keys": self._agent_net_keys,
        }

        system = self._architecture(**architecture_config)
        networks = system.create_system()

        return networks

    def variable_server(self) -> MavaVariableSource:
        """Create the variable server."""

        # Create the system networks
        networks = self.create_system()

        # Dict of variables for variable server
        variables = {}

        # Add network variables
        for net_type_key in networks.keys():
            for net_key in networks[net_type_key].keys():
                # Ensure obs and target networks are sonnet modules
                variables[f"{net_key}_{net_type_key}"] = tf2_utils.to_sonnet_module(
                    networks[net_type_key][net_key]
                ).variables

        # Add counter variables
        variables = self.create_counter_variables(variables)

        # Create variable source
        variable_source = MavaVariableSource(
            variables,
            self._checkpoint,
            self._checkpoint_subpath,
            self._checkpoint_minute_interval,
            self._termination_condition,
        )

        return variable_source

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

        # Create the system networks
        networks = self.create_system()

        # Dict of variables to retrieve from variable server
        variables = {}

        # Add the network variables
        get_keys = []
        for net_type_key in ["observations", "behaviours"]:
            for net_key in networks[net_type_key].keys():
                var_key = f"{net_key}_{net_type_key}"
                variables[var_key] = networks[net_type_key][net_key].variables
                get_keys.append(var_key)

        # Add the counter variables
        variables = self.create_counter_variables(variables)
        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
        ]
        get_keys.extend(count_names)
        counts = {name: variables[name] for name in count_names}

        variable_client = None
        if variable_source:
            # Get new variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables=variables,
                get_keys=get_keys,
                update_period=0, # always get latest variables
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.get_and_wait()

        # Pass scheduler and initialize action selectors
        exploration_schedules = {
            agent: ConstantScheduler(epsilon=0.0)
            for agent in self._environment_spec.get_agent_ids()
        }
        action_selectors_with_scheduler = initialize_epsilon_schedulers(
            exploration_schedules, networks["selectors"], self._agent_net_keys
        )

        # Create the actor which defines how we take actions.
        executor = self._executor_fn(
            observation_networks=networks["observations"],
            behaviour_networks=networks["behaviours"],
            counts=counts,
            agent_specs=self._environment_spec.get_agent_specs(),
            agent_net_keys=self._agent_net_keys,
            variable_client=variable_client,
            interval=self._evaluator_interval,
        )

        # Make the environment.
        environment = self._environment_factory(evaluation=True)  # type: ignore

        # Create logger and counter.
        evaluator_logger_config = {}
        if self._logger_config and "evaluator" in self._logger_config:
            evaluator_logger_config = self._logger_config["evaluator"]
        eval_logger = self._logger_factory(  # type: ignore
            "evaluator", **evaluator_logger_config
        )

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = self._eval_loop_fn(
            environment,
            executor,
            logger=eval_logger,
            **self._eval_loop_fn_kwargs,
        )

        # Per agent stats
        eval_loop = DetailedPerAgentStatistics(eval_loop)

        return eval_loop

    def trainer(
        self,
        variable_source: MavaVariableSource,
    ) -> mava.core.Trainer:
        """System trainer.

        Args:
            trainer_id: Id of the trainer being created.
            replay: replay data table to pull data from.
            variable_source: variable server for updating
                network variables.

        Returns:
            system trainer.
        """
        # Create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            "trainer", **trainer_logger_config
        )

        # Create the system networks
        networks = self.create_system()

        # Create the offline dataset
        environment = self._environment_factory()
        dataset = MAEnvironmentLoggerDataset(
            environment, 
            self._logdir,
            self._batch_size,
            self._shuffle_buffer_size
        )

        # Dict of variables for variable client
        variables = {}

        set_keys = []
        get_keys = []
        for net_type_key in ["observations", "behaviours"]:
            for net_key in networks[net_type_key].keys():
                variables[f"{net_key}_{net_type_key}"] = networks[net_type_key][
                    net_key
                ].variables
                set_keys.append(f"{net_key}_{net_type_key}")


        variables = self.create_counter_variables(variables)

        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
        ]
        get_keys.extend(count_names)
        counts = {name: variables[name] for name in count_names}

        # Create variable client
        variable_client = variable_utils.VariableClient(
            client=variable_source,
            variables=variables,
            get_keys=get_keys,
            set_keys=set_keys,
            update_period=10, # TODO (Claude): make variable?
        )

        # Get all the initial variables
        variable_client.get_all_and_wait()

        trainer_config: Dict[str, Any] = {
            "agents": self._agents,
            "behaviour_networks": networks["behaviours"],
            "observation_networks": networks["observations"],
            "agent_net_keys": self._agent_net_keys,
            "optimizer": self._optimizer,
            "max_gradient_norm": self._max_gradient_norm,
            "variable_client": variable_client,
            "dataset": dataset,
            "counts": counts,
            "logger": trainer_logger,
            "learning_rate_scheduler_fn": self._learning_rate_scheduler_fn,
        }

        # The trainer updates the parameters
        trainer = self._trainer_fn(**trainer_config)  # type: ignore

        trainer = ScaledDetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["value_loss"]
        )

        return trainer

    def build(self, name: str = "madqn") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name: system name.

        Returns:
            graph program for distributed system training.
        """
        program = lp.Program(name=name)

        with program.group("variable_server"):
            variable_server = program.add_node(lp.CourierNode(self.variable_server))

        with program.group("trainer"):
                program.add_node(
                    lp.CourierNode(self.trainer, variable_server)
                )

        with program.group("evaluator"):
            program.add_node(lp.CourierNode(self.evaluator, variable_server))

        return program
