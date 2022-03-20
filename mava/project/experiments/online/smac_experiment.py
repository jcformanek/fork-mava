from absl import app, flags
import functools
import wandb

from mava.utils.loggers import logger_utils
from mava.utils.environments.smac_utils import make_environment
from mava.project.systems.online import IndependentDQN, IndependentQRDQN, VDN, QMIX, QRVDN

import tensorflow as tf
import os


# WandB
wandb.init(project="Online SMAC", entity="claude_formanek")

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "map_name",
    "3m",
    "SMAC map name (str).",
)
flags.DEFINE_string(
    "algo",
    "idqn",
    "Algorithm name. One of [idqn, idqn_lambda, vdn, vdn_lambda, qmix, qmix_lambda, iqrdqn, qrvdn]",
)
flags.DEFINE_string(
    "seed",
    "1",
    "Seed mumber. Does nothing right now except repeat experiments.",
)

LAMBDA = 0.8
MAX_EXECUTOR_STEPS = 1_000_000
TRAIN_STEPS_PER_EPISODE = 2
EVALUATOR_PERIOD = 15
BATCH_SIZE=128
TARGET_UPDATE_PERIOD=200
LR = 1e-4 # TODO


def idqn(environment_factory, logger_factory, lambda_=None):
    env = environment_factory()
    max_timesteps =  env.episode_limit
    env.close()

    system = IndependentDQN(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        wandb=True,
        sequence_length=max_timesteps,
        period=max_timesteps,
        lambda_=lambda_,
        min_replay_size=BATCH_SIZE,
        batch_size=BATCH_SIZE,
        target_update_period=TARGET_UPDATE_PERIOD,
        samples_per_insert=None,
    )

    return system

def iqrdqn(environment_factory, logger_factory): # Does not support lambda
    env = environment_factory()
    max_timesteps =  env.episode_limit
    env.close()

    system = IndependentQRDQN(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        wandb=True,
        sequence_length=max_timesteps,
        period=max_timesteps,
        min_replay_size=BATCH_SIZE,
        batch_size=BATCH_SIZE,
        target_update_period=TARGET_UPDATE_PERIOD,
        samples_per_insert=None,
    )

    return system


def vdn(environment_factory, logger_factory, lambda_=None):
    env = environment_factory()
    max_timesteps =  env.episode_limit
    env.close()

    system = VDN(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        wandb=True,
        sequence_length=max_timesteps,
        period=max_timesteps,
        min_replay_size=BATCH_SIZE,
        lambda_=lambda_,
        batch_size=BATCH_SIZE,
        target_update_period=TARGET_UPDATE_PERIOD,
        samples_per_insert=None,
    )

    return system

def qmix(environment_factory, logger_factory, lambda_=None):
    env = environment_factory()
    max_timesteps =  env.episode_limit
    env.close()

    system = QMIX(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        wandb=True,
        sequence_length=max_timesteps,
        period=max_timesteps,
        min_replay_size=BATCH_SIZE,
        lambda_=lambda_,
        batch_size=BATCH_SIZE,
        target_update_period=TARGET_UPDATE_PERIOD,
        samples_per_insert=None,
    )

    return system

def qrvdn(environment_factory, logger_factory): # Does not support lambda
    env = environment_factory()
    max_timesteps =  env.episode_limit
    env.close()

    system = QRVDN(
        environment_factory=environment_factory,
        logger_factory=logger_factory,
        wandb=True,
        sequence_length=max_timesteps,
        period=max_timesteps,
        min_replay_size=BATCH_SIZE,
        batch_size=BATCH_SIZE,
        target_update_period=TARGET_UPDATE_PERIOD,
        samples_per_insert=None,
    )

    return system

def main(args):

    # Environment
    environment_factory = functools.partial(make_environment, map_name=FLAGS.map_name)

    # Logger factory
    log_every = EVALUATOR_PERIOD
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory="~/mava/",
        to_terminal=True,
        to_tensorboard=False,
        time_stamp="Dummy Logger",
        time_delta=log_every,
    )

    # Algorithm
    algo = FLAGS.algo
    if algo == "idqn":
        system = idqn(environment_factory, logger_factory)
    elif algo == "idqn_lambda":
        system = idqn(environment_factory, logger_factory, lambda_=LAMBDA)
    elif algo == "iqrdqn":
        system = iqrdqn(environment_factory, logger_factory)
    elif algo == "vdn":
        system = vdn(environment_factory, logger_factory)
    elif algo == "vdn_lambda":
        system = vdn(environment_factory, logger_factory, lambda_=LAMBDA)
    elif algo == "qmix":
        system = qmix(environment_factory, logger_factory)
    elif algo == "qmix_lambda":
        system = qmix(environment_factory, logger_factory, lambda_=LAMBDA)
    elif algo == "qrvdn":
        system = qrvdn(environment_factory, logger_factory)
    else:
        raise NotImplemented("Algo not recognised.")

    # Run system
    system.run_single_proc_system(training_steps_per_episode=TRAIN_STEPS_PER_EPISODE, evaluator_period=EVALUATOR_PERIOD, max_executor_steps=MAX_EXECUTOR_STEPS)

    return

if __name__ == "__main__":
    app.run(main)