from typing import List, Dict, Optional, Any, Sequence

import copy
import tree
import reverb
import tensorflow as tf
import sonnet as snt
import trfl
import numpy as np
from acme.tf import utils as tf2_utils
from acme.utils import loggers

from mava.utils import training_utils as train_utils

class IndependentDQNTrainer:

    def __init__(
        self,
        agents: List[str],
        q_network: snt.Module,
        optimizer: snt.Optimizer,
        discount: float,
        target_averaging: bool,
        target_update_period: int,
        target_update_rate: float,
        dataset: tf.data.Dataset,
        max_gradient_norm: float = None,
        logger: loggers.Logger = None,
    ):
        """Initialise Recurrent MADQN trainer

        Args:
            TODO
        """
        self._agents = agents
        self._logger = logger

        # Store online and target Q-networks
        self._q_network = q_network
        self._target_q_network = copy.deepcopy(q_network)

        # Other learner parameters.
        self._discount = discount

        # Set up gradient clipping
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)
        else:  # A very large number. Infinity can result in NaNs
            self._max_gradient_norm = tf.convert_to_tensor(1e10)

        # Set up target network updating
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_averaging = target_averaging
        self._target_update_period = target_update_period
        self._target_update_rate = target_update_rate

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)

        # Optimizer
        self._optimizer = optimizer

        # Expose the network variables.
        self._system_variables: Dict = {
            "q_network": self._target_q_network.variables,
        }

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

    def run(self) -> None:
        while True:
            self.step()

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""

        fetches = self._step()

        self._logger.write(fetches)

    # @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer step.

        Returns:
            losses
        """

        # Draw a batch of data from replay.
        sample: reverb.ReplaySample = next(self._iterator)

        # Compute loss
        self._forward(sample)

        # Compute and apply gradients
        self._backward()

        # Update the target networks
        self._update_target_networks()

        # Log losses
        return {"system_value_loss": self.value_loss}

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Get network variables

        Args:
            names: network names

        Returns:
            Dict[str, Dict[str, np.ndarray]]: network variables
        """

        return [tf2_utils.to_numpy(self._system_variables[name]) for name in names]

    # PRIVATE METHODS AND HOOKS

    def _update_target_networks(self) -> None:
        """Update the target networks.

        Using either target averaging or
        by directy copying the weights of the online networks every few steps.
        """
        # Get online variables
        online_variables, target_variables = self._get_variables_to_update()

        # Soft update
        if self._target_averaging:
            assert 0.0 < self._target_update_rate < 1.0
            tau = self._target_update_rate
            for src, dest in zip(online_variables, target_variables):
                dest.assign(dest * (1.0 - tau) + src * tau)

        # Or hard update
        elif tf.math.mod(self._num_steps, self._target_update_period) == 0:
            # Make online -> target network update ops.
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        self._num_steps.assign_add(1)

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

        # Global environment state for mixer
        if "s_t" in extras:
            global_env_state = train_utils.combine_dim(extras["s_t"])[0]
        else:
            global_env_state = None

        self.value_losses: Dict[str, tf.Tensor] = {}
        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            # Lists for stacking tensors later
            q_tm1_all_agents = []
            q_t_all_agents = []
            r_t_all_agents = []
            d_t_all_agents = []
            for agent in self._agents:

                q, _ = snt.static_unroll(
                    self._q_network,
                    observations[agent].observation,
                    core_state[agent][0],
                )
                q_tm1 = q[:-1]  # Chop off last timestep
                q_tm1 = trfl.batched_index(q_tm1, actions[agent][:-1]) # Chose agents action

                # Selected action at next timestep
                q_t_selector = q[1:] # Chop off first timestep
                a_t = self._agent_next_action_selector(q_t_selector, observations[agent].legal_actions[1:])
            
                # Values at next timestep
                q_t_value, _ = snt.static_unroll(
                    self._target_q_network,
                    observations[agent].observation,
                    target_core_state[agent][0],
                )
                q_t_value = q_t_value[1:] # Chop off first timestep

                # Get value of selected action
                q_t = self._agent_next_action_value(q_t_value, a_t)


                # Flatten out time and batch dim
                # Append to all agent lists
                q_tm1_all_agents.append(train_utils.combine_dim(q_tm1)[0])
                r_t_all_agents.append(
                    train_utils.combine_dim(
                        rewards[agent][:-1]  # Chop off last timestep
                    )[0]
                )
                discount = tf.cast(self._discount, dtype=discounts[agent].dtype) 
                d_t_all_agents.append(
                    train_utils.combine_dim(
                        discount * discounts[agent][:-1]  # Chop off last timestep
                    )[0]
                )
                q_t_all_agents.append(train_utils.combine_dim(q_t)[0])


            # Stack list of tensors into tensor with trailing agent dim
            q_tm1_all_agents = tf.stack(
                q_tm1_all_agents, axis=-1
            )  # shape=(TB, Num_Agents, ...)
            q_t_all_agents = tf.stack(
                q_t_all_agents, axis=-1
            )  # shape=(T,B, N, Num_Agents)
            r_t_all_agents = tf.stack(r_t_all_agents, axis=-1)
            d_t_all_agents = tf.stack(d_t_all_agents, axis=-1)

            # Possibly do mixing
            # NoOp in independent DQN
            (q_tm1_all_agents, 
            q_t_all_agents, 
            r_t_all_agents, 
            d_t_all_agents,
            global_env_state) = self._mixing(
                                q_tm1_all_agents, 
                                q_t_all_agents, 
                                r_t_all_agents, 
                                d_t_all_agents,
                                global_env_state
                            )

            target_all_agents = self._compute_target(r_t_all_agents, d_t_all_agents, q_t_all_agents)

            loss = self._compute_loss(q_tm1_all_agents, target_all_agents)

            # Get zero-padding mask
            zero_padding_mask, _ = train_utils.combine_dim(
                tf.cast(extras["zero_padding_mask"], dtype=loss.dtype)[:-1] # Chop of last timestep
            )

            masked_mean_loss = self._mask_and_mean_loss(loss, zero_padding_mask)

            self.value_loss = masked_mean_loss

        self.tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        # Calculate the gradients and update the Q-network

        # Get trainable variables.
        variables = self._get_trainable_variables()

        # Compute gradients.
        gradients = self.tape.gradient(self.value_loss, variables)

        # Maybe clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        train_utils.safe_del(self, "tape")

    def _get_variables_to_update(self):
        # Online variables
        online_variables = (
            *self._q_network.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
        )

        return online_variables, target_variables

    def _agent_next_action_selector(self, q_t_selector, legal_actions):
        # Legal action masking
        q_t_selector = tf.where(
            tf.cast(legal_actions, "bool"),
            q_t_selector,
            -999999999,
        )
        a_t = tf.argmax(q_t_selector, axis=-1)
        return a_t

    def _agent_next_action_value(self, q_t_value, a_t):
        return trfl.batched_index(q_t_value, a_t)

    def _mixing(
        self, 
        q_tm1_all_agents, 
        q_t_all_agents,
        r_t_all_agents, 
        d_t_all_agents,
        global_env_state
    ):
        """NoOp in independent DQN."""
        return q_tm1_all_agents, q_t_all_agents, r_t_all_agents, d_t_all_agents, global_env_state

    def _compute_target(self, r_t_all_agents, d_t_all_agents, q_t_all_agents):
        target = tf.stop_gradient(r_t_all_agents + d_t_all_agents * q_t_all_agents)
        return target

    def _compute_loss(self, q_tm1_all_agent, target_all_agents):
        loss = (q_tm1_all_agent - target_all_agents) ** 2
        return loss

    def _mask_and_mean_loss(self, loss, zero_padding_mask):
        # Zero-padding mask
        masked_loss = tf.reduce_sum(loss, axis=-1) * zero_padding_mask

        # Masked mean
        masked_mean_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(zero_padding_mask)

        return masked_mean_loss

    def _get_trainable_variables(self):
        variables = (self._q_network.trainable_variables)
        return variables