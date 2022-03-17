from typing import List, Dict, Optional

import copy
import trfl
import reverb
import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme.utils import loggers

from mava.project.utils.tf_utils import gather

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

        # Optional Q(lambda)
        self._lambda = 0.8

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

        # Get the relevant quantities
        observations = batch["observations"]
        actions = batch["actions"]
        legal_actions = batch["legals"]
        states = batch["states"]
        rewards = batch["rewards"][:, :-1] # Chop off last timestep
        env_discounts = tf.cast(batch["discounts"][:, :-1], "float32") # Chop off last timestep
        mask =  tf.cast(batch["mask"][:, :-1], "float32") # Chop off last timestep
        

        # Get initial hidden states for RNN
        initial_hidden_states = batch["hidden_states"][:,0] # Only first timestep
        initial_hidden_states = tf.reshape(initial_hidden_states, shape=(-1, initial_hidden_states.shape[-1])) # Flatten agent dim into batch dim
        hidden_states = (initial_hidden_states,)

        # Compute target Q-values
        target_qs_out = []
        for t in range(T):
            inputs = observations[:, t] # Extract observations at timestep t
            inputs = tf.reshape(inputs, shape=(-1, inputs.shape[-1])) # Flatten agent dim into batch dim
            qs, hidden_states = self._target_q_network(inputs, hidden_states)
            qs = tf.reshape(qs, shape=(B, N, qs.shape[-1]))            
            target_qs_out.append(qs)
        target_qs_out = tf.stack(target_qs_out, axis=1) # stack over time dim

        with tf.GradientTape() as tape:
            # Initial hidden states
            hidden_states = (initial_hidden_states,)

            # Compute online Q-values
            qs_out = []
            for t in range(T):
                inputs = observations[:, t] # Extract observations at timestep t
                inputs = tf.reshape(inputs, shape=(-1, inputs.shape[-1])) # Flatten agent dim into batch dim
                qs, hidden_states = self._q_network(inputs, hidden_states)
                qs = tf.reshape(qs, shape=(B, N, qs.shape[-1]))
                qs_out.append(qs)
            qs_out = tf.stack(qs_out, axis=1) # stack over time dim

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qs = self._get_chosen_action_qs(qs_out, actions) # Remove the last timestep on qs

            # Max over target Q-Values/ Double q learning
            target_max_qs = self._get_target_max_qs(qs_out, target_qs_out, legal_actions)

            # Mixing
            chosen_action_qs, target_max_qs = self._mixing(chosen_action_qs, target_max_qs, states)

            # Compute targets
            targets = self._compute_targets(rewards, env_discounts, target_max_qs[:,1:]) # Remove first timestep

            # Compute loss
            loss = self._compute_loss(chosen_action_qs[:,:-1], targets) # Remove last timestep

            # Zero-padding mask
            masked_loss = self._mask_loss(loss, mask)

        # Get trainable variables
        variables = self._get_trainable_variables()

        # Compute gradients.
        gradients = tape.gradient(masked_loss, variables)

        # Maybe clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        self._optimizer.apply(gradients, variables)

        # Get online and target variables
        online_variables, target_variables = self._get_variables_to_update()

        # Maybe update target network
        self._update_target_network(online_variables, target_variables)

        return {"system_value_loss": masked_loss}
        
    def get_variables(self, names):
        return [tf2_utils.to_numpy(self._system_variables[name]) for name in names]

    # HOOKS

    def extra_setup(self, **kwargs):
        """Noop"""
        return

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
        all_hidden_states = []
        for agent in self._agents:
            all_observations.append(observations[agent].observation)
            all_legals.append(observations[agent].legal_actions)
            all_actions.append(actions[agent])
            all_rewards.append(rewards[agent])
            all_discounts.append(discounts[agent])
            all_hidden_states.append(extras["core_states"][agent][0][0]) # TODO why are states nested here

        all_observations = tf.stack(all_observations, axis=-2) # (B,T,N,O)
        all_legals = tf.stack(all_legals, axis=-2) # (B,T,N,A)
        all_hidden_states = tf.stack(all_hidden_states, axis=-2) # (B,T,N, hidden_dim)
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
            "hidden_states": all_hidden_states,
            "mask": mask,
            "states": states
        }

        return batch

    def _get_chosen_action_qs(self, qs_out, actions):
        return gather(qs_out, actions, axis=3, keepdims=False)

    def _get_target_max_qs(self, qs_out, target_qs_out, legal_actions):
        qs_out_selector = tf.where(legal_actions, qs_out, -9999999) # legal action masking
        cur_max_actions = tf.argmax(qs_out_selector, axis=3)
        target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1)
        return target_max_qs

    def _mixing(
        self, 
        chosen_action_qs, 
        target_max_qs,
        states
    ):
        """NoOp in independent DQN."""
        return chosen_action_qs, target_max_qs

    def _compute_targets(self, rewards, env_discounts, target_max_qs):
        if self._lambda is not None:
            # Duplicate rewards and discount for all agents
            rewards = tf.concat([rewards]*3, axis=-1)
            env_discounts = tf.concat([env_discounts]*3, axis=-1)

            # Make time major for trfl
            rewards = tf.transpose(rewards, perm=[1,0,2])
            env_discounts = tf.transpose(env_discounts, perm=[1,0,2])
            target_max_qs = tf.transpose(target_max_qs, perm=[1,0,2])

            # Get time and batch dim
            T = rewards.shape[0]
            B = rewards.shape[1]

            # Flatten agent dim into batch-dim
            rewards = tf.reshape(rewards, shape=(T, -1))
            env_discounts = tf.reshape(env_discounts, shape=(T, -1))
            target_max_qs = tf.reshape(target_max_qs, shape=(T, -1))

            # Q(lambda)
            targets = trfl.multistep_forward_view(
                rewards,
                self._discount * env_discounts,
                target_max_qs,
                lambda_=self._lambda,
                back_prop=False
            )
            # Unpack agent dim again
            targets = tf.reshape(targets, shape=(T, B, -1))

            # Make batch major again
            targets = tf.transpose(targets, perm=[1,0,2])
        else:
            targets = rewards + self._discount * env_discounts * target_max_qs
        return tf.stop_gradient(targets)

    def _compute_loss(self, chosen_actions_qs, targets):
        td_error = (chosen_actions_qs - targets)
        loss = 0.5 * tf.square(td_error)
        return loss

    def _mask_loss(self, loss, mask):
        # Zero-padding mask
        masked_loss = loss * mask

        # Masked mean
        masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

        return masked_loss

    def _get_trainable_variables(self):
        variables = (self._q_network.trainable_variables)
        return variables

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