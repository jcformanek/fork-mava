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
"""Example running MADQN on SMAC with multiple trainers."""

import functools
from datetime import datetime
from typing import Any

import launchpad as lp
from absl import app, flags

from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.systems.tf import madqn
from mava.utils import enums, lp_utils
from mava.utils.enums import ArchitectureType
from mava.utils.environments import smac_utils
from mava.utils.loggers import logger_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "env_name",
    "3m",
    "SMAC map name.",
)

flags.DEFINE_string(
    "mava_id",
    str(datetime.now()),
    "Experiment identifier that can be used to continue experiments.",
)

flags.DEFINE_string("base_dir", "~/mava", "Base dir to store experiments.")


def main(_: Any) -> None:
    """Run MADQN on SMAC with multiple trainers."""

    # Environment.
    environment_factory = functools.partial(
        smac_utils.make_environment,
        map_name=FLAGS.env_name,
    )

    # Networks.
    network_factory = lp_utils.partial_kwargs(
        madqn.make_default_networks, architecture_type=ArchitectureType.recurrent
    )

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
    program = madqn.MADQN(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        num_executors=1,
        exploration_scheduler_fn=LinearExplorationScheduler(
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=5e-6,
        ),
        shared_weights=False,
        trainer_networks=enums.Trainer.one_trainer_per_network,
        network_sampling_setup=enums.NetworkSampler.fixed_agent_networks,
        trainer_fn=madqn.MADQNRecurrentTrainer,
        executor_fn=madqn.MADQNRecurrentExecutor,
        max_replay_size=5000,
        min_replay_size=32,
        batch_size=32,
        samples_per_insert=4,
        evaluator_interval={"executor_episodes": 2000},
        checkpoint_subpath=checkpoint_dir,
    ).build()

    # Only the trainer should use the GPU (if available)
    local_resources = lp_utils.to_device(
        program_nodes=program.groups.keys(), nodes_on_gpu=["trainer"]
    )

    # Launch.
    lp.launch(
        program,
        lp.LaunchType.LOCAL_MULTI_PROCESSING,
        terminal="current_terminal",
        local_resources=local_resources,
    )


if __name__ == "__main__":
    app.run(main)
