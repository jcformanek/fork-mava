import dm_env
import numpy as np

from mava.offline.offline_utils import MAEnvironmentLogger, MAEnvironmentLoggerDataset
from mava.utils.environments.smac_utils import make_environment

env = make_environment()
env = MAEnvironmentLogger(env, 100, trajectories_per_file=100)
NUM_EPISODES = 501

timestep, _ = env.reset()

i = 0
while True:

    actions = {"agent_0": np.array(1), "agent_1": np.array(1), "agent_2": np.array(1)}
    for agent in actions.keys():
        if timestep.observation[agent].legal_actions[0] == 1:
            actions[agent] = np.array(0)

    timestep, _ = env.step(actions)

    if timestep.step_type == dm_env.StepType.LAST:
        i += 1

    if i == NUM_EPISODES:
        break

dataset = MAEnvironmentLoggerDataset(env, "./environment_logs")

sample = next(iter(dataset))

print(sample)

print("Done")
