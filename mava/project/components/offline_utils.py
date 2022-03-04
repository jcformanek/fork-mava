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

class OfflineTrajectory(NamedTuple):
    observations: Dict
    actions: Dict
    rewards: Dict
    discounts: Dict
    legal_actions: Dict
    extras: Dict


class MAEnvironmentLogger:
    def __init__(
        self,
        environment,
        max_trajectory_length: int,
        logdir: str = "./environment_logs",
        trajectories_per_file: int = 100_000,
    ):
        self._environment = environment
        environment_spec = MAEnvironmentSpec(environment)
        self._agents = environment_spec.get_agent_ids()
        schema = _get_schema(environment_spec)

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


class MAEnvironmentLoggerDataset:
    def __init__(self, environment, logdir, batch_size=32, shuffle_buffer_size=1000):
        self._environment = environment
        self._spec = MAEnvironmentSpec(environment)
        self._schema = _get_schema(self._spec)
        self._agents = self._spec.get_agent_ids()

        file_path = os.path.join(logdir, "*.tfrecord")
        filenames = glob.glob(file_path)
        filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)

        self._dataset = (
            filename_dataset.interleave(
                lambda x: tf.data.TFRecordDataset(x).map(self._decode_fn),
                cycle_length=None,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False,
                block_length=None
            )
            .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=False)
            .batch(batch_size)
            .repeat()
        )

    def _decode_fn(self, record_bytes):
        example = tf.io.parse_single_example(
            record_bytes,
            tree.map_structure(
                lambda x: tf.io.FixedLenFeature([], dtype=tf.string), self._schema
            ),
        )

        for key, item in self._schema.items():
            example[key] = tf.io.parse_tensor(example[key], item.dtype)

        observations = {}
        actions = {}
        rewards = {}
        discounts = {}
        legal_actions = {}
        extras = {}
        for agent in self._agents:
            observations[agent] = example[agent + "observations"]
            legal_actions[agent]=example[agent + "legal_actions"]
            actions[agent] = example[agent + "actions"]
            rewards[agent] = example[agent + "rewards"]
            discounts[agent] = example[agent + "discounts"]

        ## Extras
        # Zero padding
        extras["zero_padding_mask"] = example["zero_padding_mask"]
        
        # Global env state
        if "env_state" in example:
            extras["env_state"] = example["env_state"]

        offline_replay_sample = OfflineTrajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            discounts=discounts,
            legal_actions=legal_actions,
            extras=extras
        )

        return offline_replay_sample

    def __iter__(self):
        return self._dataset.__iter__()

    def __next__(self):
        return self._dataset.__next__()

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
            return getattr(self._dataset, name)


def _get_schema(environment_spec: MAEnvironmentSpec):
    agent_specs = environment_spec.get_agent_specs()

    schema = {}
    for agent in environment_spec.get_agent_ids():
        spec = agent_specs[agent]

        schema[agent + "observations"] = spec.observations.observation
        schema[agent + "legal_actions"] = spec.observations.legal_actions
        schema[agent + "actions"] = spec.actions
        schema[agent + "rewards"] = spec.rewards
        schema[agent + "discounts"] = spec.discounts
    
    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = np.array(1, dtype=np.float32)

    # Global env state
    extras_spec = environment_spec.get_extra_specs()
    if "s_t" in extras_spec:
        schema["env_state"] = extras_spec["s_t"]

    return schema
