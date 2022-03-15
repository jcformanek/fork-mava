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
from mava.project.systems.online.value_based.independent_dqn import IndependentDQNTrainer

class IndependentOffPGTrainer(IndependentDQNTrainer):

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
            agents,
            q_network,
            optimizer,
            discount,
            target_averaging,
            target_update_period,
            target_update_rate,
            dataset,
            max_gradient_norm,
            logger
        )

        # Initialized during extras setup
        self._policy_network = None
        self._policy_optimizer = None
        self._mixer = None
        self._target_mixer = None
        self._td_lambda = None
        self._q_optimizer = None

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
            hidden_states = batch["hidden_states"]
            hidden_states = tf.reshape(hidden_states, shape=(1, -1, hidden_states.shape[-1])) # Flatten agent dim into batch dim

            # Unroll the policy
            logits_out = []
            for t in range(T):
                inputs = observations[:, t] # Extract observations at timestep t
                inputs = tf.reshape(inputs, shape=(-1, inputs.shape[-1])) # Flatten agent dim into batch dim
                logits, hidden_states = self._policy_forward(inputs, hidden_states)
                logits = tf.reshape(logits, shape=(B, N, logits.shape[-1]))       
                logits_out.append(logits)
            logits_out = tf.stack(logits_out, axis=1) # stack over time dim

            # Mask illegal actions
            # logits_out = tf.where(legal_actions, logits_out, -1e8)
            probs_out = tf.nn.softmax(logits_out, axis=-1)
            # probs_out = tf.where(legal_actions, probs_out, 0.0)
            # probs_out = probs_out / tf.reduce_sum(probs_out, axis=-1, keepdims=True)

            action_values = gather(q_vals, actions)
            baseline = tf.reduce_sum(probs_out * q_vals, axis=-1)
            advantage = action_values - baseline
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits_out)
            coe = self._mixer.k(states + 1e-8)  # add small number so that zero padded elements don't cause div by zero error
                                                # they get masked out later

            pg_loss = coe * cross_entropy * advantage

            # Zero-padding masking
            pg_mask = tf.concat([mask] * N, axis=2)
            pg_loss = tf.where(tf.cast(pg_mask, "bool"), pg_loss, 0.0)
            pg_loss = tf.reduce_sum(pg_loss) / tf.reduce_sum(pg_mask)

            # Critic learning
            q_vals = gather(q_vals, actions, axis=-1)
            target_q_vals = gather(target_q_vals, actions, axis=-1)

            # Mixing critics
            q_vals = self._mixer(q_vals, states)
            target_q_vals = self._target_mixer(target_q_vals, states)

            # Make time major for trfl
            rewards = tf.transpose(rewards, perm=[1,0,2])
            env_discounts = tf.transpose(env_discounts, perm=[1,0,2])
            target_q_vals = tf.transpose(target_q_vals, perm=[1,0,2])

            # Q(lambda)
            target = trfl.multistep_forward_view(
                tf.squeeze(rewards[:-1,:]),
                tf.squeeze(self._discount * env_discounts[:-1,:]),
                tf.squeeze(target_q_vals[1:,:]),
                lambda_=0.8,
                back_prop=False
            )
            # Make batch major again
            target = tf.transpose(target, perm=[1,0])

            td_error = tf.stop_gradient(target) - tf.squeeze(q_vals[:,:-1])
            q_loss = 0.5 * tf.square(td_error)

            # Masking 
            q_loss = q_loss * tf.squeeze(mask[:,:-1])
            q_loss = tf.reduce_sum(q_loss) / tf.reduce_sum(mask[:,:-1])

        # Apply gradients Q-network and Mixer
        variables = (*self._q_network.trainable_variables, *self._mixer.trainable_variables) # Get trainable variables
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

    # HOOKS

    def extra_setup(self, **kwargs):
        # Store policy _network and optimizer
        self._policy_network = kwargs["policy_network"]
        self._policy_optimizer = kwargs["policy_optimizer"]

        # Overwrite system variables, executor only needs policy_variables
        self._system_variables: Dict = {
            "policy_network": self._policy_network.variables,
        }

        # Setup mixers
        self._mixer = QMixer(len(self._agents))
        self._target_mixer = QMixer(len(self._agents))

        # Store TD Lambda
        self._td_lambda = kwargs["lambda_"]

        # Make this link for readability
        self._q_optimizer = self._optimizer
        

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