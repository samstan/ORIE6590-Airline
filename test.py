from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import numpy as np
import airline
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from torch import nn

def evaluate(model, env, numiters):
    rewards = []
    for j in range(numiters):
        obs = env.reset()
        tot_reward = 0
        for i in range(env.tau):
            action, states = model.predict(obs)
            _, reward, _, _ = env.step(action)
            tot_reward += reward
        rewards.append(tot_reward)
    return np.mean(rewards), np.std(rewards)

# Model is the random policy
def policy(env):
    return env.action_space.sample()

A = np.asarray([[1, 1, 0,0,0,0], [ 0,0, 1, 1, 1, 1], [ 0,0, 0,0, 1, 1] ])
tau = 23
P = np.ones((tau, A.shape[1]))/3
c = [5, 5, 5]
f = range(10, 16)
CONFIG = {'A': A, 'f': f, 'P': P, 'starting_state': c , 'tau': tau}

env = gym.make('Airline-v0', config=CONFIG)

check_env(env)

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(2,last_layer_dim_pi),
            nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


model = A2C(CustomActorCriticPolicy, env, verbose=1)
model.learn(5000)