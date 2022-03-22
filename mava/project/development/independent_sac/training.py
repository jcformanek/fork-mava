from typing import List, Dict, Optional

import numpy as np
import copy
import trfl
import reverb
import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import tensorflow_probability as tfp
from mava.project.components.mixers import QMixer

from mava.utils import training_utils as train_utils
from mava.project.utils.tf_utils import gather

class IndependentSACTrainer:

    def __init__(
        self,
        agents: List[str],
        num_actions: int,
        critic_network: snt.Module,
        policy_network: snt.Module,
        critic_optimizer: snt.Optimizer,
        policy_optimizer: snt.Optimizer,
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
        self._critic_network_1 = critic_network
        self._target_critic_network_1 = copy.deepcopy(critic_network)
        self._critic_network_2 = copy.deepcopy(critic_network)
        self._target_critic_network_2 = copy.deepcopy(critic_network)

        # Store online and target policy networks
        self._policy_network = policy_network

        # Other learner parameters.
        self._discount = discount

        # Set up gradient clipping
        if max_gradient_norm is not None:
            self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm, "float32")
        else:  # A very large number. Infinity can result in NaNs
            self._max_gradient_norm = tf.convert_to_tensor(1e10, "float32")

        # Set up target network updating
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_averaging = target_averaging
        self._target_update_period = target_update_period
        self._target_update_rate = target_update_rate

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)

        # Optimizers
        self._policy_optimizer = policy_optimizer
        self._critic_optimizer_1 = critic_optimizer
        self._critic_optimizer_2 = copy.deepcopy(critic_optimizer)

        # Automatic temperature tuning term
        self._target_entropy = -np.log((1.0 / num_actions)) * 0.98
        self._log_alpha = tf.Variable(0, trainable=True, dtype="float32")
        self._alpha_optimizer = snt.optimizers.Adam(learning_rate=1e-4)

        # Expose the network variables.
        self._system_variables: Dict = {
            "policy_network": self._policy_network.variables,
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
        # fetches = self._step()
        fetches = self._step()

        self._logger.write(fetches)

        return fetches

    @tf.function
    def _step(self):
        # Sample replay
        inputs: reverb.ReplaySample = next(self._iterator)

        # Batch agent inputs together
        batch = self._batch_inputs(inputs)

        # Get max sequence length, batch size and num agents
        B = batch["observations"].shape[0]
        T = batch["observations"].shape[1]
        N = len(self._agents)
        A = batch["legals"].shape[-1]

        # Get the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"]
        env_discounts = tf.cast(batch["discounts"], "float32")
        mask =  tf.cast(batch["mask"], "float32")
        mask = tf.stack([mask]*N, axis=2)
        mask = tf.reshape(mask, shape=(mask.shape[:-1]))
        with tf.GradientTape(persistent=True) as tape: 

            # Get initial hidden states for RNN
            hidden_states = self._policy_network.initial_state(B*N) # Flatten agent dim into batch dim

            # Unroll the policy
            logits_out = []
            for t in range(T):
                inputs = observations[:, t] # Extract observations at timestep t
                inputs = tf.reshape(inputs, shape=(-1, inputs.shape[-1])) # Flatten agent dim into batch dim
                logits, hidden_states = self._policy_network(inputs, hidden_states)
                logits = tf.reshape(logits, shape=(B, N, logits.shape[-1]))       
                logits_out.append(logits)
            logits_out = tf.stack(logits_out, axis=1) # stack over time dim shape=[B,T,N,A]

            # Mask illegal actions
            logits_out = tf.where(legal_actions, logits_out, -1e8)
            probs = tf.nn.softmax(logits_out, axis=-1)

            # Have to deal with situation of 0.0 probabilities because we can't do log 0
            z = tf.cast(tf.where(legal_actions, 0.0, 1e-8), probs.dtype)
            log_probs = tf.math.log(probs + z)

            # Q-vals
            q_vals_1, q_vals_2, target_q_vals_1, target_q_vals_2 = self._critic_forward(observations, states)

            # Min Q-vals
            min_q_vals = tf.minimum(q_vals_1, q_vals_2)

            # Policy loss
            inside_term = tf.exp(self._log_alpha) * log_probs - min_q_vals
            policy_loss = tf.reduce_sum(probs * inside_term, axis=-1)
            policy_loss = tf.reduce_sum(policy_loss * mask) / tf.reduce_sum(mask)

            # Critic learning
            min_target_q_vals = probs * (tf.minimum(target_q_vals_1, target_q_vals_2) - tf.exp(self._log_alpha) * log_probs)
            min_target_q_vals = tf.reduce_sum(min_target_q_vals, axis=-1)
            target = rewards[:,:-1] + self._discount * env_discounts[:,:-1] * min_target_q_vals[:,1:]
            target = tf.stop_gradient(target)

            q1 = gather(q_vals_1, actions)[:,:-1]
            q2 = gather(q_vals_2, actions)[:,:-1]
            critic_loss_1 = tf.reduce_sum(0.5 * tf.square(target - q1) * mask[:,:-1]) / tf.reduce_sum(mask[:,:-1])
            critic_loss_2 = tf.reduce_sum(0.5 * tf.square(target - q2) * mask[:,:-1]) / tf.reduce_sum(mask[:,:-1])

            # Temperature loss
            log_action_probabilities = tf.reduce_sum(log_probs * probs, axis=-1)
            alpha_loss = - tf.reduce_sum(self._log_alpha * tf.stop_gradient(log_action_probabilities + self._target_entropy) * mask) / tf.reduce_sum(mask)

        # Apply gradients Q-network 1
        variables = self._critic_network_1.trainable_variables # Get trainable variables
        gradients = tape.gradient(critic_loss_1, variables) # Compute gradients.        
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0] # Maybe clip gradients.
        self._critic_optimizer_1.apply(gradients, variables) # Apply gradients.

        # Apply gradients Q-network 2
        variables = self._critic_network_2.trainable_variables # Get trainable variables
        gradients = tape.gradient(critic_loss_2, variables) # Compute gradients.        
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0] # Maybe clip gradients.
        self._critic_optimizer_2.apply(gradients, variables) # Apply gradients.

        # Apply gradients policy network
        variables = self._policy_network.trainable_variables # Get trainable variables
        gradients = tape.gradient(policy_loss, variables) # Compute gradients.        
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0] # Maybe clip gradients.
        self._policy_optimizer.apply(gradients, variables) # Apply gradients.

        # Apply gradients temperature param
        variables = (self._log_alpha,) # Get trainable variables
        gradients = tape.gradient(alpha_loss, variables) # Compute gradients.        
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0] # Maybe clip gradients.
        self._alpha_optimizer.apply(gradients, variables) # Apply gradients.

        # Get online and target variables
        online_variables, target_variables = self._get_variables_to_update()

        # Maybe update target network
        self._update_target_network(online_variables, target_variables)

        # Delete gradient tape
        train_utils.safe_del(self, "tape")

        return {"critic_loss_1": critic_loss_1, "critic_loss_2": critic_loss_2, "policy_loss": policy_loss, "log_alpha": self._log_alpha}
        
    def get_variables(self, names):
        return [tf2_utils.to_numpy(self._system_variables[name]) for name in names]

    # HOOKS

    def _batch_inputs(self, inputs):
        # Unpack inputs
        data = inputs.data
        observations, actions, rewards, discounts, _, extras = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.start_of_episode,
            data.extras,
        )

        all_observations = []
        all_legals = []
        all_actions = []
        all_rewards = []
        all_discounts = []
        for agent in self._agents:
            all_observations.append(observations[agent].observation)
            all_legals.append(observations[agent].legal_actions)
            all_actions.append(actions[agent])
            all_rewards.append(rewards[agent])
            all_discounts.append(discounts[agent])

        all_observations = tf.stack(all_observations, axis=-2) # (B,T,N,O)
        all_legals = tf.stack(all_legals, axis=-2) # (B,T,N,A)
        all_actions = tf.stack(all_actions, axis=-1) # (B,T,N,1)
        all_rewards = tf.reduce_mean(tf.stack(all_rewards, axis=-1), axis=-1, keepdims=True) # (B,T,1)
        all_discounts = tf.reduce_mean(tf.stack(all_discounts, axis=-1), axis=-1, keepdims=True) # (B,T,1)

        # Cast legals to bool
        all_legals = tf.cast(all_legals, "bool")

        mask = tf.expand_dims(extras["zero_padding_mask"], axis=-1) # (B,T,1)

        states = extras["s_t"] if "s_t" in extras else None # (B,T,N,S)

        batch = {
            "observations": all_observations,
            "actions": all_actions,
            "rewards": all_rewards,
            "discounts": all_discounts,
            "legals": all_legals,
            "mask": mask,
            "states": states
        }

        return batch

    def _policy_forward(self, observations, hidden_states, legal_actions):
        """Step policy forward by one timestep."""
        logits, hidden_states = self._policy_network(observations, hidden_states)

        # Mask illegal actions
        logits = tf.where(legal_actions, logits, -1e8)
        probs = tf.nn.softmax(logits, axis=-1)
        probs = tf.where(legal_actions, probs, 0)
        probs = probs / tf.reduce_sum(probs, axis=-1)

        # Distribution
        dist = tfp.distributions.Categorical(probs=probs)

        # Sampled actions
        actions = dist.sample()

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        
        return agent_outs

    def _critic_forward(self, observations, states):
        N = observations.shape[2] # num agents
        extra_state_info = [states] * N
        extra_state_info = tf.stack(extra_state_info, axis=2)
        critic_inputs = tf.concat([observations, extra_state_info], axis=-1)

        q_vals_1 = self._critic_network_1(critic_inputs)   
        q_vals_2 = self._critic_network_2(critic_inputs)

        # Target Q-vals
        target_q_vals_1 = self._target_critic_network_1(critic_inputs)   
        target_q_vals_2 = self._target_critic_network_2(critic_inputs)

        return q_vals_1, q_vals_2, target_q_vals_1, target_q_vals_2

    
    def _get_variables_to_update(self):
        # Online variables
        online_variables = (
            *self._critic_network_1.variables,
            *self._critic_network_2.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_critic_network_1.variables,
            *self._target_critic_network_2.variables,
        )

        return online_variables, target_variables

    def _update_target_network(self, online_variables, target_variables) -> None:
        """Update the target networks.

        Using either target averaging or
        by directy copying the weights of the online networks every few steps.
        """
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