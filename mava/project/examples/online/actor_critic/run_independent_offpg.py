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
"""Example running recurent MADQN on SMAC."""


import functools
from datetime import datetime
from typing import Any

import sonnet as snt
from absl import app, flags

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationTimestepScheduler,
)
from mava.project.systems.online.actor_critic.independent_offpg import IndependentOffPG
from mava.utils.environments.smac_utils import make_environment
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "map_name",
    "3m",
    "Starcraft 2 micromanagement map name (str).",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)
flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def main(_: Any) -> None:
    """Example running recurrent MADQN on SMAC environment."""

    # Environment
    environment_factory = functools.partial(make_environment, map_name=FLAGS.map_name)

    # Checkpointer appends "Checkpoints" to checkpoint_dir
    checkpoint_dir = f"{FLAGS.base_dir}/{FLAGS.mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=FLAGS.base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=FLAGS.mava_id,
        time_delta=log_every,
    )

    # Distributed program
    program = IndependentOffPG(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        q_optimizer=snt.optimizers.Adam(1e-4),
        policy_optimizer=snt.optimizers.Adam(1e-4),
        exploration_scheduler=LinearExplorationTimestepScheduler(
            epsilon_start=0.5, epsilon_min=0.05, epsilon_decay_steps=500_000,
        ),
        checkpoint_subpath=checkpoint_dir,
        batch_size=10,
        sequence_length=61,
        period=61,
        min_replay_size=10,
        target_update_period=600,
        max_gradient_norm=20.0,
        samples_per_insert=None,
    )

    program._samples_per_insert = None
    program.run_single_proc_system()

    # program = program.build()

    # # Only the trainer should use the GPU (if available)
    # local_resources = lp_utils.to_device(
    #     program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
    # )

    # # Launch
    # lp.launch(
    #     program,
    #     lp.LaunchType.LOCAL_MULTI_PROCESSING,
    #     terminal="current_terminal",
    #     local_resources=local_resources,
    # )


if __name__ == "__main__":
    app.run(main)
