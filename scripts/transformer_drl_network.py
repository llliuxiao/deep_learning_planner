from typing import Callable

import gymnasium as gym
import torch
from einops import reduce
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from transformer_network import Transformer, PositionalEncoding
from utils.parameters import *


class CustomMlpExtractor(nn.Module):
    def __init__(self,
                 feature_dim,
                 last_layer_dim_pi: int = 256,
                 last_layer_dim_vf: int = 128):
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.mlp_extractor_actor = nn.Sequential(nn.Linear(feature_dim, self.latent_dim_pi), nn.ReLU())
        self.mlp_extractor_critic = nn.Sequential(nn.Linear(feature_dim, self.latent_dim_vf), nn.ReLU())

    def forward(self, tensor):
        return self.mlp_extractor_actor(tensor), self.mlp_extractor_critic(tensor)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp_extractor_actor(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp_extractor_critic(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor(feature_dim=self.features_dim)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.latent_dim_pi = 256
        self.latent_dim_vf = 128
        self.laser_mask = torch.triu(torch.ones((laser_length, laser_length), dtype=torch.bool), 1).to("cuda")
        # laser
        self.laser_pre = nn.Linear(1080, 512)
        self.laser_position_encoding = PositionalEncoding(d_model=512, max_seq_len=6)
        self.laser_transformer = Transformer(dim=512, dim_head=64,
                                             heads=8, depth=6,
                                             attn_dropout=0.1, ff_dropout=0.1)

        # global plan
        self.global_plan_pre = nn.Linear(3, 256)
        self.global_plan_position_encoding = PositionalEncoding(d_model=256, max_seq_len=20)
        self.global_plan_transformer = Transformer(dim=256, dim_head=64,
                                                   heads=4, depth=4,
                                                   attn_dropout=0.1, ff_dropout=0.1)
        self.dense = nn.Sequential(
            nn.Linear(512 + 256 + 3, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU()
        )

    def forward(self, obs):
        laser = obs["laser"]
        global_plan = obs["global_plan"]
        goal = obs["goal"]
        pooled_laser = self.laser_pre(laser)
        positional_laser = self.laser_position_encoding(pooled_laser)
        attended_laser = self.laser_transformer(positional_laser, attn_mask=self.laser_mask)

        high_dim_global_plan = self.global_plan_pre(global_plan)
        positional_path = self.global_plan_position_encoding(high_dim_global_plan)
        attended_global_plan = self.global_plan_transformer(positional_path)

        laser_token = reduce(attended_laser, 'b f d -> b d', 'mean')
        global_plan_token = reduce(attended_global_plan, 'b f d -> b d', 'mean')

        tensor = torch.concat((laser_token, global_plan_token, goal), dim=1)
        feature = self.dense(tensor)
        return feature
