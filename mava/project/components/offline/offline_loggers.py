import glob
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple

import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree

from mava.adders.reverb.base import Trajectory
from mava.specs import MAEnvironmentSpec
from mava.types import OLT
from mava.project.components.offline.offline_utils import get_schema

class MAOfflineEnvironmentLogger:
    def __init__(
        self,
        environment,
        max_trajectory_length: int,
        logdir: str = "./offline_env_logs",
        trajectories_per_file: int = 1000,
    ):
        self._environment = environment
        environment_spec = MAEnvironmentSpec(environment)
        self._agents = environment_spec.get_agent_ids()
        schema = get_schema(environment_spec)

        self._buffer = [
            tree.map_structure(
                lambda x: np.zeros(
                    dtype=x.dtype, shape=(max_trajectory_length, *x.shape)
                ),
                schema,
            )
            for _ in range(trajectories_per_file)
        ]

        self._trajectories_per_file = trajectories_per_file

        self._logdir = logdir
        if not os.path.exists(logdir):
            # Make dir if not exist
            os.makedirs(logdir)
        self._num_writes = 0

        self._max_trajectory_length = max_trajectory_length
        self._timestep: Optional[dm_env.TimeStep] = None
        self._extras: Optional[Dict] = None

        self._ctr = 0
        self._t = 0

    def reset(self) -> Tuple[dm_env.TimeStep, Dict]:
        """Resets the env and log the first timestep.

        Returns:
            dm.env timestep, extras
        """
        timestep = self._environment.reset()

        if isinstance(timestep, tuple):
            self._timestep, self._extras = timestep
        else:
            self._extras = {}

        return self._timestep, self._extras

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dm_env.TimeStep, Dict]:
        """Steps the env and logs timestep.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm.env timestep, extras
        """
        next_timestep = self._environment.step(actions)

        if isinstance(next_timestep, tuple) and len(next_timestep) == 2:
            next_timestep, next_extras = next_timestep
        else:
            next_extras = {}

        # Log current timestep
        self._log_timestep(self._timestep, self._extras, next_timestep, actions)

        # If next_timestep is the final one, then write.
        if self._ctr == self._trajectories_per_file:
            self._write()

        # Update current timestep and extras
        self._timestep = next_timestep
        self._extras = next_extras

        return self._timestep, self._extras

    def _log_timestep(
        self, timestep: dm_env.TimeStep, extras: Dict, next_timestep: dm_env.TimeStep, actions: Dict
    ) -> None:
        assert self._t < self._max_trajectory_length, "Max time steps exceeded."
        assert (
            self._ctr < self._trajectories_per_file
        ), "Buffer full. Make sure the buffer was flushed after write."

        for agent in self._agents:
            self._buffer[self._ctr][agent + "observations"][
                self._t
            ] = timestep.observation[agent].observation

            self._buffer[self._ctr][agent + "legal_actions"][
                self._t
            ] = timestep.observation[agent].legal_actions

            self._buffer[self._ctr][agent + "actions"][self._t] = actions[agent]

            self._buffer[self._ctr][agent + "rewards"][
                self._t
            ] = next_timestep.reward[agent]

            self._buffer[self._ctr][agent + "discounts"][
                self._t
            ] = next_timestep.discount[agent]
            
        ## Extras
        # Zero padding mask
        self._buffer[self._ctr]["zero_padding_mask"][
            self._t
        ] = np.array(1, dtype=np.float32)

        # Global env state
        if "s_t" in extras:
            self._buffer[self._ctr]["env_state"][
                self._t
            ] = extras["s_t"]

        if next_timestep.step_type == dm_env.StepType.LAST:
            # Maybe zero pad sequence
            while self._t < self._max_trajectory_length:
                for agent in self._agents:
                    for item in ["observations", "legal_actions", "actions", "rewards", "discounts"]:
                        self._buffer[self._ctr][agent + item][
                            self._t
                        ] = np.zeros_like(
                            self._buffer[self._ctr][agent + item][0]
                        )

                    ## Extras
                    # Zero-padding mask
                    self._buffer[self._ctr]["zero_padding_mask"][
                        self._t
                    ] = np.zeros_like(
                        self._buffer[self._ctr]["zero_padding_mask"][0]
                    )

                    # Global env state
                    if "env_state" in self._buffer:
                        self._buffer[self._ctr]["env_state"][
                            self._t
                        ] = np.zeros_like(
                            self._buffer[self._ctr]["env_state"][0]
                        )

                # Increment time
                self._t += 1

            # Increment buffer counter
            self._ctr += 1

            # Zero timestep
            self._t = 0
        else:
            # Increment timestep
            self._t += 1

    def _write(self) -> None:
        assert self._ctr == self._trajectories_per_file, "Buffer is not full yet."

        filename = os.path.join(
            self._logdir, f"trajectory_log_{self._num_writes}.tfrecord"
        )
        with tf.io.TFRecordWriter(filename) as file_writer:
            for trajectory in self._buffer:

                # Convert numpy to tf.train features
                dict_of_features = tree.map_structure(
                    self._numpy_to_feature, trajectory
                )

                # Create Example for writing
                features_for_example = tf.train.Features(feature=dict_of_features)
                example = tf.train.Example(features=features_for_example)

                # Write to file
                file_writer.write(example.SerializeToString())

        # Increment write counter
        self._num_writes += 1

        # Flush buffer and reset ctr
        self._buffer = tree.map_structure(lambda x: np.zeros_like(x), self._buffer)
        self._ctr = 0

    def _numpy_to_feature(self, np_array: np.ndarray):
        tensor = tf.convert_to_tensor(np_array)
        serialized_tensor = tf.io.serialize_tensor(tensor)
        bytes_list = tf.train.BytesList(value=[serialized_tensor.numpy()])
        feature_of_bytes = tf.train.Feature(bytes_list=bytes_list)
        return feature_of_bytes

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)