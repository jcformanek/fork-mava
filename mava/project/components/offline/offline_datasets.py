import glob
import os
from typing import Any

import numpy as np
import reverb
import tensorflow as tf
import tree

from mava.specs import MAEnvironmentSpec
from mava.types import OLT
from mava.project.components.offline.offline_utils import get_schema
from mava.adders.reverb.base import Step

class MAOfflineEnvironmentDataset:
    def __init__(self, environment, logdir, batch_size=32, shuffle_buffer_size=1000):
        self._environment = environment
        self._spec = MAEnvironmentSpec(environment)
        self._schema = get_schema(self._spec)
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

        # Make OLTs
        for agent in self._agents:
            observations[agent] = OLT(
                observation=observations[agent],
                legal_actions=legal_actions[agent],
                terminal=tf.zeros(1, dtype="float32") # TODO only a place holder for now
            )

        ## Extras
        # Zero padding
        zero_padding_mask = example["zero_padding_mask"]
        extras["zero_padding_mask"] = zero_padding_mask
        # Global env state
        if "env_state" in example:
            extras["s_t"] = example["env_state"]

        # Start of episode
        start_of_episode = tf.zeros(1, dtype="float32") # TODO only a place holder for now

        # Pack into Step
        reverb_sample_data = Step(
            observations = observations,
            actions = actions,
            rewards = rewards,
            discounts = discounts,
            start_of_episode = start_of_episode,
            extras = extras
        )

        # Make reverb sample so that interface same as in online algos
        reverb_sample_info = reverb.SampleInfo(key=-1, probability=-1.0, table_size=-1, priority=-1.0) # TODO only a place holder for now

        # Rever sample
        reverb_sample = reverb.ReplaySample(info=reverb_sample_info, data=reverb_sample_data)

        return reverb_sample

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