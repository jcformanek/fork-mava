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

"""Value Decomposition trainer implementation."""
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import reverb
import sonnet as snt
import tensorflow as tf
import tree
import numpy as np
import trfl
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from mava.systems.tf.dist_madqn.execution import DistMADQNExecutor

from mava.systems.tf.madqn.training import MADQNRecurrentTrainer
from mava.systems.tf.variable_utils import VariableClient
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class DistMADQNTrainer(MADQNRecurrentTrainer):
    """Value Decomposition Trainer.

    This is the trainer component of a Value Decomposition system.
    IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        value_networks: Dict[str, snt.Module],
        target_value_networks: Dict[str, snt.Module],
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise Value Decompostion trainer.

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            value_networks: value networks for each agent in
                the system.
            target_value_networks: target value networks.
            optimizer: optimizer(s) for updating value networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            value_networks=value_networks,
            target_value_networks=target_value_networks,
            optimizer=optimizer,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            variable_client=variable_client,
            counts=counts,
            agent_net_keys=agent_net_keys,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )

        self._atoms = None

    def setup_dist_atoms(self, atoms: tf.Tensor) -> None:
        """Initialize the mixer network.

        Args:
            mixer: mixer network
            mixer_optimizer: optimizer for updating mixing networks.
        """
        self._atoms = atoms

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass.

        Args:
            inputs: input data from the data table (transitions)
        """
        # Convert to time major
        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

        # Note (dries): The unused variable is start_of_episodes.
        observations, actions, rewards, discounts, _, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.start_of_episode,
            data.extras,
        )

        # Get initial state for the LSTM from replay and
        # extract the first state in the sequence.
        core_state = tree.map_structure(lambda s: s[0, :, :], extras["core_states"])
        target_core_state = tree.map_structure(
            lambda s: s[0, :, :], extras["core_states"]
        )

        # TODO (dries): Take out all the data_points that does not need
        #  to be processed here at the start. Therefore it does not have
        #  to be done later on and saves processing time.

        self.value_losses: Dict[str, tf.Tensor] = {}

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Note (dries): We are assuming that only the policy network
            # is recurrent and not the observation network.
            obs_trans, target_obs_trans = self._transform_observations(observations)

            for agent in self._trainer_agent_list:
                agent_key = self._agent_net_keys[agent]

                # Double Q-learning
                q_logits, _ = snt.static_unroll(
                    self._value_networks[agent_key],
                    obs_trans[agent],
                    core_state[agent][0],
                )
                q_logits = tf.reshape(q_logits, shape=(*q_logits.shape[:-1], -1, 51)) # TODO don't hardcode
                logits_q_tm1 = q_logits[:-1]  # Chop off last timestep
            
                logits_q_t, _ = snt.static_unroll(
                    self._target_value_networks[agent_key],
                    target_obs_trans[agent],
                    target_core_state[agent][0],
                )
                logits_q_t = tf.reshape(logits_q_t, shape=(*logits_q_t.shape[:-1], -1, 51)) # TODO don't hardcode
                logits_q_t = logits_q_t[1:] # Chop off first timestep

                q_probs = tf.nn.softmax(q_logits)
                q_mean = tf.reduce_sum(q_probs * self._atoms, -1)
                q_t_selector = q_mean[1:]  # Chop off first timestep

                # Legal action masking
                q_t_selector = tf.where(
                    tf.cast(observations[agent].legal_actions[1:], "bool"),
                    q_t_selector,
                    -999999999,
                )

                # Flatten out time and batch dim
                logits_q_tm1, _ = train_utils.combine_dim(logits_q_tm1)
                q_t_selector, _ = train_utils.combine_dim(q_t_selector)
                logits_q_t, _ = train_utils.combine_dim(logits_q_t)
                a_tm1, _ = train_utils.combine_dim(
                    actions[agent][:-1]  # Chop off last timestep
                )
                r_t, _ = train_utils.combine_dim(
                    rewards[agent][:-1]  # Chop off last timestep
                )
                d_t, _ = train_utils.combine_dim(
                    discounts[agent][:-1]  # Chop off last timestep
                )

                # Cast the additional discount to match
                # the environment discount dtype.
                discount = tf.cast(self._discount, dtype=discounts[agent].dtype)

                # Value loss
                value_loss, _ = trfl.categorical_dist_double_qlearning(
                    self._atoms,
                    logits_q_tm1,
                    a_tm1,
                    r_t,
                    d_t * discount,
                    self._atoms,
                    logits_q_t,
                    q_t_selector,
                )   

                # Zero-padding mask
                zero_padding_mask, _ = train_utils.combine_dim(
                    tf.cast(extras["zero_padding_mask"], dtype=value_loss.dtype)[:-1]
                )
                masked_loss = value_loss * zero_padding_mask
                self.value_losses[agent] = tf.reduce_sum(masked_loss) / tf.reduce_sum(
                    zero_padding_mask
                )

        self.tape = tape


class QRDQNTrainer(DistMADQNTrainer):
    """Value Decomposition Trainer.

    This is the trainer component of a Value Decomposition system.
    IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        value_networks: Dict[str, snt.Module],
        target_value_networks: Dict[str, snt.Module],
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        observation_networks: Dict[str, snt.Module],
        target_observation_networks: Dict[str, snt.Module],
        variable_client: VariableClient,
        counts: Dict[str, Any],
        agent_net_keys: Dict[str, str],
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
        learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]] = None,
    ):
        """Initialise Value Decompostion trainer.

        Args:
            agents: agent ids, e.g. "agent_0".
            agent_types: agent types, e.g. "speaker" or "listener".
            value_networks: value networks for each agent in
                the system.
            target_value_networks: target value networks.
            optimizer: optimizer(s) for updating value networks.
            discount: discount factor for TD updates.
            target_averaging: whether to use polyak averaging for target network
                updates.
            target_update_period: number of steps before target networks are
                updated.
            target_update_rate: update rate when using averaging.
            dataset: training dataset.
            observation_networks: network for feature
                extraction from raw observation.
            target_observation_networks: target observation
                network.
            variable_client: The client used to manage the variables.
            counts: step counter object.
            agent_net_keys: specifies what network each agent uses.
            max_gradient_norm: maximum allowed norm for gradients
                before clipping is applied.
            logger: logger object for logging trainer
                statistics.
            learning_rate_scheduler_fn: dict with two functions (one for the policy and
                one for the critic optimizer), that takes in a trainer step t and
                returns the current learning rate.
        """

        super().__init__(
            agents=agents,
            agent_types=agent_types,
            value_networks=value_networks,
            target_value_networks=target_value_networks,
            optimizer=optimizer,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            observation_networks=observation_networks,
            target_observation_networks=target_observation_networks,
            variable_client=variable_client,
            counts=counts,
            agent_net_keys=agent_net_keys,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
        )

        self._atoms = None
        self._tau = None
        self.huber_loss = tf.compat.v1.losses.huber_loss

    def setup_dist_atoms(self, atoms: tf.Tensor) -> None:
        """Initialize the mixer network.

        Args:
            mixer: mixer network
            mixer_optimizer: optimizer for updating mixing networks.
        """
        self._atoms = atoms
        self._tau = np.array([(2*(i-1)+1)/(2*self._atoms) for i in range(1, self._atoms+1)])

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass.

        Args:
            inputs: input data from the data table (transitions)
        """
        # Convert to time major
        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

        # Note (dries): The unused variable is start_of_episodes.
        observations, actions, rewards, discounts, _, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.start_of_episode,
            data.extras,
        )

        # Get initial state for the LSTM from replay and
        # extract the first state in the sequence.
        core_state = tree.map_structure(lambda s: s[0, :, :], extras["core_states"])
        target_core_state = tree.map_structure(
            lambda s: s[0, :, :], extras["core_states"]
        )

        self.value_losses: Dict[str, tf.Tensor] = {}
        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # Note (dries): We are assuming that only the policy network
            # is recurrent and not the observation network.
            obs_trans, target_obs_trans = self._transform_observations(observations)

            for agent in self._trainer_agent_list:
                agent_key = self._agent_net_keys[agent]

                # Double Q-learning
                q, _ = snt.static_unroll(
                    self._value_networks[agent_key],
                    obs_trans[agent],
                    core_state[agent][0],
                )
                q = tf.reshape(q, shape=(*q.shape[:-1], -1, 51)) # TODO don't hardcode
                q_tm1 = q[:-1]  # Chop off last timestep
            
                q_t, _ = snt.static_unroll(
                    self._target_value_networks[agent_key],
                    target_obs_trans[agent],
                    target_core_state[agent][0],
                )
                q_t = tf.reshape(q_t, shape=(*q_t.shape[:-1], -1, 51)) # TODO don't hardcode
                q_t = q_t[1:] # Chop off first timestep

                q_t_selector = tf.reduce_mean(q, -1)[1:]  # Chop off first timestep
                # Legal action masking
                q_t_selector = tf.where(
                    tf.cast(observations[agent].legal_actions[1:], "bool"),
                    q_t_selector,
                    -999999999,
                )
                a_t = tf.argmax(q_t_selector, axis=-1)

                # Flatten out time and batch dim
                q_tm1, _ = train_utils.combine_dim(q_tm1)
                a_tm1, _ = train_utils.combine_dim(
                    actions[agent][:-1]  # Chop off last timestep
                )
                r_t, _ = train_utils.combine_dim(
                    rewards[agent][:-1]  # Chop off last timestep
                )
                d_t, _ = train_utils.combine_dim(
                    discounts[agent][:-1]  # Chop off last timestep
                )
                a_t, _ = train_utils.combine_dim(a_t)
                q_t, _ = train_utils.combine_dim(q_t)

                # Cast the additional discount to match
                # the environment discount dtype.
                discount = tf.cast(self._discount, dtype=discounts[agent].dtype)

                # Quantile Huber loss
                # See https://github.com/marload/DistRL-TensorFlow2/blob/master/QR-DQN/QR-DQN.py
                one_hot_action_indices = tf.expand_dims(tf.one_hot(a_t, q_t.shape[-2], dtype=q_t.dtype), axis=-1) 
                target = tf.expand_dims(r_t, axis=-1) + discount * tf.expand_dims(d_t, axis=-1) * tf.reduce_sum(q_t * one_hot_action_indices, axis=-2, keepdims=False)
                target = tf.stop_gradient(target)
                pred = q_tm1
                pred = tf.reduce_sum(pred * tf.expand_dims(tf.one_hot(a_tm1, depth=q_t.shape[-2], dtype='float32'), -1), axis=1) # TODO num_actions
                pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, self._atoms])
                target_tile = tf.tile(tf.expand_dims(target, axis=1), [1, self._atoms, 1])
                huber_loss = self.huber_loss(target_tile, pred_tile)
                tau = tf.cast(tf.reshape(self._tau, [1, self._atoms]), dtype='float32')
                inv_tau = 1.0 - tau
                tau = tf.tile(tf.expand_dims(tau, axis=1), [1, self._atoms, 1])
                inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, self._atoms, 1])
                error_loss = tf.math.subtract(target_tile, pred_tile)
                loss = tf.where(tf.less(error_loss, 0.0), inv_tau * huber_loss, tau * huber_loss)
                loss = tf.reduce_sum(tf.reduce_mean(loss, axis=2), axis=1)

                # Zero-padding mask
                zero_padding_mask, _ = train_utils.combine_dim(
                    tf.cast(extras["zero_padding_mask"], dtype=loss.dtype)[:-1]
                )
                masked_loss = loss * zero_padding_mask
                self.value_losses[agent] = tf.reduce_sum(masked_loss) / tf.reduce_sum(
                    zero_padding_mask
                )

        self.tape = tape