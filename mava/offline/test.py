from mava.offline.offline_utils import MAEnvironmentLoggerDataset
from mava.utils.environments.debugging_utils import make_environment

env = make_environment()

dataset = MAEnvironmentLoggerDataset(env, "./environment_logs", shuffle_buffer_size=500)
dataset = iter(dataset)

i = 0
while True:
    print(i)
    print(next(dataset))
    break
