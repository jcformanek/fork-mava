from typing import List, Dict, Optional, Any, Sequence

import tensorflow as tf
import sonnet as snt
import numpy as np
from acme.utils import loggers

from mava.project.systems.online.value_based.independent_dqn import IndependentDQNTrainer
from mava.project.utils.tf_utils import gather
from mava.project.components.mixers import QMixer

class QMIXTrainer(IndependentDQNTrainer):

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

    # HOOKS

    def extra_setup(self, **kwargs):
        self._mixer = QMixer(len(self._agents), embed_dim=kwargs["embed_dim"], hypernet_embed=kwargs["hypernet_embed_dim"])
        self._target_mixer = QMixer(len(self._agents), embed_dim=kwargs["embed_dim"], hypernet_embed=kwargs["hypernet_embed_dim"])

    def _mixing(
        self, 
        chosen_action_qs, 
        target_max_qs,
        states
    ):
        """NoOp in independent DQN."""
        chosen_action_qs = self._mixer(chosen_action_qs, states)
        target_max_qs = self._target_mixer(target_max_qs, states)
        return chosen_action_qs, target_max_qs

    def _get_trainable_variables(self):
        variables = (*self._q_network.trainable_variables, *self._mixer.trainable_variables)
        return variables

    def _get_variables_to_update(self):
        # Online variables
        online_variables = (
            *self._q_network.variables,
            *self._mixer.variables
        )

        # Get target variables
        target_variables = (
            *self._target_q_network.variables,
            *self._target_mixer.variables
        )

        return online_variables, target_variables