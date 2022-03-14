from typing import List, Dict, Optional

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

class IndependentOffPGTrainer:

    def __init__(
        self,
        agents: List[str],
        q_network: snt.Module,
        policy_network: snt.Module,
        q_optimizer: snt.Optimizer,
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
        self._q_network = q_network
        self._target_q_network = copy.deepcopy(q_network)

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
        self._q_optimizer = q_optimizer

        # Expose the network variables.
        self._system_variables: Dict = {
            "policy_network": self._policy_network.variables,
        }

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp: Optional[float] = None

        self._mask_before_softmax = True
        self._mixer = QMixer(len(self._agents))
        self._target_mixer = QMixer(len(self._agents))
        self._td_lambda = 0.8

    def run(self) -> None:
        while True:
            self.step()

    def step(self) -> None:
        """Trainer step to update the parameters of the agents in the system"""
        # fetches = self._step()
        fetches = self._step()

        self._logger.write(fetches)

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
        with tf.GradientTape(persistent=True) as tape:

            # Forward pass through Q-networks
            q_vals, target_q_vals = self._critic_forward(observations, states)     

            # Get initial hidden states for RNN
            hidden_states = self._policy_network.initial_state(B*N) # Flatten agent dim into batch dim

            # Unroll the policy
            logits_out = []
            for t in range(T):
                inputs = observations[:, t] # Extract observations at timestep t
                inputs = tf.reshape(inputs, shape=(-1, inputs.shape[-1])) # Flatten agent dim into batch dim
                logits, hidden_states = self._policy_forward(inputs, hidden_states)
                logits = tf.reshape(logits, shape=(B, N, logits.shape[-1]))       
                logits_out.append(logits)
            logits_out = tf.stack(logits_out, axis=1) # stack over time dim

            probs_out = tf.nn.softmax(logits_out, axis=-1)

            action_values = gather(q_vals, actions)
            baseline = tf.reduce_sum(probs_out * q_vals, axis=-1)
            advantage = action_values - baseline
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits_out)
            coe = self._mixer.k(states + 0.0000001)
            pg_loss = coe * cross_entropy * advantage

            # Masking
            pg_mask = tf.concat([mask] * N, axis=2)
            pg_loss = tf.where(tf.cast(pg_mask, "bool"), pg_loss, 0.0)
            pg_loss = tf.reduce_sum(pg_loss) / tf.reduce_sum(pg_mask)

            # Critic learning
            q_vals = gather(q_vals, actions, axis=-1)
            target_q_vals = gather(target_q_vals, actions, axis=-1)

            # Mixing critics
            q_vals = self._mixer(q_vals, states)
            target_q_vals = self._target_mixer(target_q_vals, states)

            rewards = tf.transpose(rewards, perm=[1,0,2])
            env_discounts = tf.transpose(env_discounts, perm=[1,0,2])
            target_q_vals = tf.transpose(target_q_vals, perm=[1,0,2])
            target = trfl.generalized_lambda_returns(
                tf.squeeze(rewards),
                tf.squeeze(self._discount * env_discounts),
                tf.squeeze(target_q_vals),
                bootstrap_value=tf.squeeze(tf.zeros_like(target_q_vals[0])),
                lambda_=0.8
            )
            target = tf.transpose(target, perm=[1,0])

            td_error = tf.stop_gradient(target[:,1:]) - tf.squeeze(q_vals[:,:-1])
            q_loss = 0.5 * tf.square(td_error)

            # Masking 
            q_loss = q_loss * tf.squeeze(mask[:,:-1])
            q_loss = tf.reduce_sum(q_loss) / tf.reduce_sum(mask[:,:-1])

        # Apply gradients Q-network
        variables = self._q_network.trainable_variables # Get trainable variables
        gradients = tape.gradient(q_loss, variables) # Compute gradients.        
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0] # Maybe clip gradients.
        self._q_optimizer.apply(gradients, variables) # Apply gradients.

        # Apply gradients policy network
        variables = self._policy_network.trainable_variables # Get trainable variables
        gradients = tape.gradient(pg_loss, variables) # Compute gradients.        
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0] # Maybe clip gradients.
        self._policy_optimizer.apply(gradients, variables) # Apply gradients.

        # Get online and target variables
        online_variables, target_variables = self._get_variables_to_update()

        # Maybe update target network
        self._update_target_network(online_variables, target_variables)

        # Delete gradient tape
        train_utils.safe_del(self, "tape")

        return {"q_loss": q_loss, "policy_loss": pg_loss}
        
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

    def _policy_forward(self, observations, hidden_states):
        """Step policy forward by one timestep."""
        agent_outs, hidden_states = self._policy_network(observations, hidden_states)
        
        return agent_outs, hidden_states

    def _critic_forward(self, observations, states):
        N = observations.shape[2] # num agents
        extra_state_info = [states] * N
        extra_state_info = tf.stack(extra_state_info, axis=2)
        critic_inputs = tf.concat([observations, extra_state_info], axis=-1)

        q_vals = self._q_network(critic_inputs)   
        target_q_vals = self._target_q_network(critic_inputs)

        return q_vals, target_q_vals

    
    def _get_variables_to_update(self):
        # Online variables
        online_variables = (
            *self._q_network.variables,
            *self._mixer.variables,
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._target_mixer.variables
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