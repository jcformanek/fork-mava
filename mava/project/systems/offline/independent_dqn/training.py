from typing import List, Dict, Optional

import copy
import trfl
import reverb
import tensorflow as tf
import sonnet as snt
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from mava.project.systems.online.independent_dqn.training import IndependentDQNTrainer

from mava.project.utils.tf_utils import gather

class BatchConstrainedIndependentDQNTrainer(IndependentDQNTrainer):

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
        super().__init__(
            agents=agents,
            q_network=q_network,
            optimizer=optimizer,
            discount=discount,
            target_averaging=target_averaging,
            target_update_period=target_update_period,
            target_update_rate=target_update_rate,
            dataset=dataset,
            max_gradient_norm=max_gradient_norm,
            logger=logger,
        )

    @tf.function
    def _step(self, inputs):

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
        mask =  tf.cast(batch["mask"], "float32")
        

        # Get initial hidden states for RNN
        if batch["hidden_states"] is not None:
            initial_hidden_states = batch["hidden_states"][:,0] # Only first timestep
            initial_hidden_states = tf.reshape(initial_hidden_states, shape=(-1, initial_hidden_states.shape[-1])) # Flatten agent dim into batch dim
        else:
            initial_hidden_states = self._q_network.initial_state(B*N)[0]
        # Pack into a tuple because thats what sonnet expects
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

        with tf.GradientTape(persistent=True) as tape:
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

            # Unroll behaviour cloning network
            probs_out = []
            hidden_states = self._behaviour_cloning_network.initial_state(B*N)
            for t in range(T):
                inputs = observations[:, t] # Extract observations at timestep t
                inputs = tf.reshape(inputs, shape=(-1, inputs.shape[-1])) # Flatten agent dim into batch dim
                probs, hidden_states = self._behaviour_cloning_network(inputs, hidden_states)
                probs = tf.reshape(probs, shape=(B, N, probs.shape[-1]))
                probs_out.append(probs)
            probs_out = tf.stack(probs_out, axis=1) # stack over time dim

            # Legal action masking
            probs_out = probs_out * tf.cast(legal_actions, "float32")
            probs_out_sum = tf.reduce_sum(probs_out, axis=-1, keepdims=True)
            probs_out_sum_is_zero = probs_out_sum == 0.0 
            probs_out_sum = tf.where(probs_out_sum_is_zero, 1.0, probs_out_sum) # Note if probs out sum is zero then this is a zero padded example
            probs_out = probs_out / probs_out_sum
            probs_out = tf.where(probs_out_sum_is_zero, 1/probs_out.shape[-1], probs_out)

            # Behaviour cloning loss
            one_hot_actions = tf.one_hot(actions, depth=probs_out.shape[-1], axis=-1)
            bc_loss = tf.keras.metrics.categorical_crossentropy(one_hot_actions, probs_out)
            bc_mask = tf.concat([mask]*N, axis=-1)
            bc_loss = tf.reduce_sum(bc_loss * bc_mask) / tf.reduce_sum(bc_mask)

            # Max over target Q-Values/ Double q learning
            target_max_qs = self._get_target_max_qs(qs_out, target_qs_out, legal_actions, probs_out)

            # Mixing
            chosen_action_qs, target_max_qs = self._mixing(chosen_action_qs, target_max_qs, states)

            # Compute targets
            targets = self._compute_targets(rewards, env_discounts, target_max_qs[:,1:]) # Remove first timestep

            # Compute loss
            loss = self._compute_loss(chosen_action_qs[:,:-1], targets) # Remove last timestep

            # Zero-padding mask
            masked_loss = self._mask_loss(loss, mask[:,:-1]) # Remove last timestep

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

        # Behaviour cloning update
        variables = self._behaviour_cloning_network.trainable_variables
        gradients = tape.gradient(bc_loss, variables)
        self._behaviour_cloning_optimizer.apply(gradients, variables)

        del tape

        return {"system_value_loss": masked_loss, "behaviour_cloning_loss": bc_loss}

    # HOOKS

    def extra_setup(self, **kwargs):
        """Maybe Q-lambda"""
        self._lambda = kwargs["lambda_"]
        self._threshold = kwargs["threshold"]
        self._behaviour_cloning_network = kwargs["behaviour_cloning_network"]
        self._behaviour_cloning_optimizer = kwargs["behaviour_cloning_optimizer"]
        return

    def _get_target_max_qs(self, qs_out, target_qs_out, legal_actions, behaviour_clonning_probs):
        bc_probs_greater_than_threshold = behaviour_clonning_probs >= self._threshold
        valid_bc_actions = tf.reduce_sum(tf.cast(bc_probs_greater_than_threshold, dtype='float32'), axis=-1, keepdims=True) > 0
        action_mask = tf.where(valid_bc_actions, bc_probs_greater_than_threshold, legal_actions)

        qs_out_selector = tf.where(action_mask, qs_out, -9999999) # legal action masking
        cur_max_actions = tf.argmax(qs_out_selector, axis=3)
        target_max_qs = gather(target_qs_out, cur_max_actions, axis=-1)
        return target_max_qs