import random

import launchpad as lp

# from launchpad.launch import worker_manager
import reverb
import sonnet as snt

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf import madqn
from mava.utils.environments.meltingpot_utils.env_utils import EnvironmentFactory
from mava.utils.environments.meltingpot_utils.registry import (
    get_system_creator_cls,
    register_evaluator_creator,
    register_focal_networks_setter,
    register_networks_restorer,
    register_system_creator_cls,
)


@register_evaluator_creator("MADQN")
def create_evaluator(system):
    replay = system.replay()
    counter = system.counter(False)
    trainer = system.trainer(replay, counter)
    return system.evaluator(trainer, counter, trainer)._environment_loop


@register_focal_networks_setter("MADQN")
def madqn_focal_networks_setter(evaluator, networks):
    trained_network_keys = list(networks.keys())
    for key in evaluator._q_networks.keys():
        idx = random.randint(0, len(trained_network_keys) - 1)
        evaluator._q_networks[key] = networks[trained_network_keys[idx]]


def create_replay_client(priority_tables):
    server = reverb.Server(tables=priority_tables, checkpointer=None)
    client = reverb.Client(f"localhost:{server.port}")
    return client


@register_networks_restorer("MADQN")
def restore_madqn_networks(substrate, network_factory, checkpoint_dir):
    substrate_environment_factory = EnvironmentFactory(substrate=substrate)
    system_creator_cls = get_system_creator_cls("MADQN")
    substrate_system_creator = system_creator_cls(
        substrate_environment_factory, network_factory, None, checkpoint_dir
    )
    system = substrate_system_creator()
    import pdb

    pdb.set_trace()
    replay = lp.ReverbNode(system.replay).create_handle()
    # replay = create_replay_client(system.replay())
    counter = system.counter(False)
    trainer = system.trainer(replay, counter)
    return trainer._q_network


@register_system_creator_cls("MADQN")
class MADQNSystemCreator:
    def __init__(
        self, environment_factory, network_factory, logger_factory, checkpoint_dir
    ) -> None:
        self.system = madqn.MADQN(
            environment_factory=environment_factory,
            network_factory=network_factory,
            logger_factory=logger_factory,
            num_executors=1,
            exploration_scheduler_fn=LinearExplorationScheduler(
                epsilon_min=0.05, epsilon_decay=1e-4
            ),
            importance_sampling_exponent=0.2,
            optimizer=snt.optimizers.Adam(learning_rate=1e-4),
            checkpoint_subpath=checkpoint_dir,
        )

    def __call__(self):
        return self.system
