from argparse import Namespace
from typing import Literal

import torch
from pydantic import BaseModel
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.algo.utils.torch_utils import calc_num_elements

from sample_factory.utils.utils import log

from torch import nn as nn


class EncoderConfig(BaseModel):
    """
    Configuration for an encoder.

    Args:
        extra_fc_layers (int): Number of extra fully connected (fc) layers. Default is 0.
        num_filters (int): Number of filters. Default is 64.
        num_res_blocks (int): Number of residual blocks. Default is 1.
        activation_func (Literal['ReLU', 'ELU']): Activation function to use. Default is 'ReLU'.
        hidden_size (int): Hidden size for extra fc layers. Default is 128.
    """
    extra_fc_layers: int = 0
    num_filters: int = 64
    num_res_blocks: int = 1
    activation_func: Literal['ReLU', 'ELU', 'Mish'] = 'ReLU'
    hidden_size: int = 128


def activation_func(cfg: EncoderConfig) -> nn.Module:
    """
    Returns an instance of nn.Module representing the activation function specified in the configuration.

    Args:
        cfg (EncoderConfig): Encoder configuration.

    Returns:
        nn.Module: Instance of nn.Module representing the activation function.

    Raises:
        Exception: If the activation function specified in the configuration is unknown.
    """
    if cfg.activation_func == "ELU":
        return nn.ELU(inplace=True)
    elif cfg.activation_func == "ReLU":
        return nn.ReLU(inplace=True)
    elif cfg.activation_func == "Mish":
        return nn.Mish(inplace=True)
    else:
        raise Exception("Unknown activation_func")


class ResBlock(nn.Module):
    """
    Residual block in the encoder.

    Args:
        cfg (EncoderConfig): Encoder configuration.
        input_ch (int): Input channel size.
        output_ch (int): Output channel size.
    """

    def __init__(self, cfg: EncoderConfig, input_ch, output_ch):
        super().__init__()

        layers = [
            activation_func(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),
            activation_func(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    """
    ResNet-based encoder.

    Args:
        cfg (Config): Configuration.
        obs_space (ObsSpace): Observation space.
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.encoder_cfg: EncoderConfig = EncoderConfig(**cfg.encoder)

        input_ch = obs_space['obs'].shape[0]
        resnet_conf = [[self.encoder_cfg.num_filters, self.encoder_cfg.num_res_blocks]]
        curr_input_channels = input_ch
        layers = []

        for out_channels, res_blocks in resnet_conf:
            layers.extend([nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)])
            layers.extend([ResBlock(self.encoder_cfg, out_channels, out_channels) for _ in range(res_blocks)])
            curr_input_channels = out_channels

        layers.append(activation_func(self.encoder_cfg))
        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space['obs'].shape)
        self.encoder_out_size = self.conv_head_out_size

        if self.encoder_cfg.extra_fc_layers:
            self.extra_linear = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.encoder_cfg.hidden_size),
                activation_func(self.encoder_cfg),
            )
            self.encoder_out_size = self.encoder_cfg.hidden_size

        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, x):
        x = x['obs']
        x = self.conv_head(x)
        x = x.contiguous().view(-1, self.conv_head_out_size)

        if self.encoder_cfg.extra_fc_layers:
            x = self.extra_linear(x)

        return x


def main():
    exp_cfg = {'encoder': EncoderConfig().dict()}
    r = 5
    obs = torch.rand(1, 3, r * 2 + 1, r * 2 + 1)
    q_obs = {'obs': obs}
    # noinspection PyTypeChecker
    re = ResnetEncoder(Namespace(**exp_cfg), dict(obs=obs[0]))
    re(q_obs)


if __name__ == '__main__':
    main()
