from typing import Dict, List, Optional, Sequence, Union

import sonnet as snt
import tensorflow as tf

from mava import specs as mava_specs
from mava.components.tf import networks
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy
from mava.utils.enums import ArchitectureType, Network


class MeltingPotConvNet(snt.Module):
    def __init__(
        self,
        conv_out_channels: List[int] = [16, 32],
        conv_kernel_sizes: List[int] = [8, 4],
        conv_strides: List[int] = [8, 1],
        fc_layers: List[int] = [64, 64],
        name: Optional[str] = None,
    ) -> None:
        """Convolution network used to train agents in meltingpot

        Args:
            conv_out_channels (List[int], optional): output channels for the conv
            layers. Defaults to [16, 32].
            conv_kernel_sizes (List[int], optional): kernel size for the conv layers.
            Defaults to [8, 4].
            conv_strides (List[int], optional): strides for the conv layers.
            Defaults to [8, 1].
            fc_layers (List[int], optional): outputs nodes for the fc layers.
            Defaults to [64, 64].
            name (Optional[str], optional): name. Defaults to None.
        """
        super(MeltingPotConvNet, self).__init__(name=name)
        conv_layers = []
        fc_layers = []
        for output_channels, kernel_shape, stride in zip(
            conv_out_channels, conv_kernel_sizes, conv_strides
        ):
            conv_layer = snt.Conv2D(
                output_channels=output_channels,
                kernel_shape=kernel_shape,
                stride=stride,
            )
            conv_layers += [conv_layer]
            conv_layers += [tf.nn.relu]
        for nodes in fc_layers:
            fc_layer = snt.Linear(nodes)
            fc_layers += [fc_layer]
            fc_layers += [tf.nn.relu]
        all_layers = conv_layers + [snt.Flatten()] + fc_layers
        self._network = snt.Sequential(all_layers)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._network(inputs)


def make_default_madqn_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
):
    specs = environment_spec.get_agent_specs()
    specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    q_networks = {}
    action_selectors = {}
    for key in specs.keys():
        num_dimensions = specs[key].actions.num_values
        network = snt.Sequential([MeltingPotConvNet(), snt.Linear(num_dimensions)])
        q_networks[key] = network
        action_selectors[key] = EpsilonGreedy

    return {
        "q_networks": q_networks,
        "action_selectors": action_selectors,
    }
