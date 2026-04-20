""" some hardcoded data/constants + utility functions for RL env setup """

from functools import partial

import doy
import gym
import numpy as np
import matplotlib.pyplot as plt
# from procgen import ProcgenEnv

from RL.env import load_environment


class RealtimeViewer():
    def __init__(self):

        self.fig = None
        self.panels = None



    def initialize(self, next_obs, method = 'wide'):
        plt.ion()

        if method == 'wide':
            fig, ax = plt.subplots(nrows=1, ncols=6)
            img0 = ax[0].imshow(next_obs[0].permute(1, 2, 0).cpu())
            img1 = ax[1].imshow(next_obs[0].permute(1, 2, 0).cpu())
            img2 = ax[2].imshow(next_obs[0].permute(1, 2, 0).cpu())
            img3 = ax[3].imshow(next_obs[0].permute(1, 2, 0).cpu())
            img4 = ax[4].imshow(next_obs[0].permute(1, 2, 0).cpu())
            img5 = ax[5].imshow(next_obs[0].permute(1, 2, 0).cpu())

            ax[0].set_title('t-2')
            ax[1].set_title('t-1')
            ax[2].set_title('t')
            ax[3].set_title('t+1')
            ax[4].set_title('t+2')
            ax[5].set_title('t+3')

            self.panels = {
                '0': img0,
                '1': img1,
                '2': img2,
                '3': img3,
                '4': img4,
                '5': img5,
            }

            self.fig = fig


        plt.show()

    def min_max_norm(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def update(self, display_list):

        for i in range(len(display_list)):
            self.panels[str(i)].set_data(self.min_max_norm(display_list[i]).detach().squeeze().cpu().permute(1, 2, 0))
        
        
        # self.panels['1'].set_data(self.min_max_norm(wm.buf_obs[1][0]).detach().squeeze().cpu().permute(1, 2, 0))
        # self.panels['2'].set_data(self.min_max_norm(wm.buf_obs[2][0]).detach().squeeze().cpu().permute(1, 2, 0))
        # self.panels['3'].set_data(self.min_max_norm(wm.buf_imagine[-1][0][0]).detach().squeeze().cpu().permute(1, 2, 0))
        # self.panels['4'].set_data(self.min_max_norm(wm.buf_imagine[-1][1][0]).detach().squeeze().cpu().permute(1, 2, 0))
        # self.panels['5'].set_data(self.min_max_norm(wm.buf_imagine[-1][2][0]).detach().squeeze().cpu().permute(1, 2, 0))
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()



def normalize_return(ep_ret, env_name):
    """normalizes returns based on URP and expert returns above"""
    return doy.normalize_into_range(
        lower=urp_ep_return[env_name],
        upper=expert_ep_return[env_name],
        v=ep_ret,
    )


def setup_gym_env(env_id, num_envs, render_kwargs):
    # env = load_environment(env_id=env_id, num_envs=num_envs, **render_kwargs)
    env = gym.make(
            env_id,
            num_envs=num_envs,
            render_mode = "rgb_array",
            **render_kwargs,
        )
    return env

# def setup_procgen_env(num_envs, env_id, gamma,  distribution_mode='easy', render_mode='rgb_array'):
#     envs = ProcgenEnv(
#         num_envs=num_envs,
#         env_name=env_id,
#         num_levels=0,
#         start_level=0,
#         distribution_mode=distribution_mode,
#         render_mode=render_mode
#     )

#     envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
#     envs.single_action_space = envs.action_space
#     envs.single_observation_space = envs.observation_space["rgb"]
#     envs.is_vector_env = True
#     envs = gym.wrappers.RecordEpisodeStatistics(envs)
#     envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
#     envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
#     assert isinstance(
#         envs.single_action_space, gym.spaces.Discrete
#     ), "only discrete action space is supported"

#     envs.normalize_return = partial(normalize_return, env_name=env_id)
#     return envs


ta_dim = {
    "bigfish": 15,
    "bossfight": 15,
    "caveflyer": 15,
    "chaser": 15,
    "climber": 15,
    "coinrun": 15,
    "dodgeball": 15,
    "fruitbot": 15,
    "heist": 15,
    "jumper": 15,
    "leaper": 15,
    "maze": 15,
    "miner": 15,
    "ninja": 15,
    "plunder": 15,
    "starpilot": 15,
    "walker": 6
}

procgen_names = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
    "walker"
]


procgen_action_meanings = np.array(
    [
        "LEFT-DOWN",
        "LEFT",
        "LEFT-UP",
        "DOWN",
        "NOOP",
        "UP",
        "RIGHT-DOWN",
        "RIGHT",
        "RIGHT-UP",
        "D",
        "A",
        "W",
        "S",
        "Q",
        "E",
    ]
)

# mean episodic returns for procgen (easy) under uniform random policy
urp_ep_return = {
    "bigfish": 0.8742888,
    "bossfight": 0.04618272,
    "caveflyer": 2.5970738,
    "chaser": 0.6632482,
    "climber": 2.2242901,
    "coinrun": 2.704834,
    "dodgeball": 0.612983,
    "fruitbot": -2.5330205,
    "heist": 3.0758767,
    "jumper": 3.4105318,
    "leaper": 2.5105245,
    "maze": 4.2726293,
    "miner": 1.2513667,
    "ninja": 2.5599792,
    "plunder": 4.3207445,
    "starpilot": 1.5251881,
}

# mean episodic returns for procgen (easy) under expert policy (from expert data)
expert_ep_return = {
    "bigfish": 36.336166,
    "bossfight": 11.634365,
    "caveflyer": 9.183605,
    "chaser": 9.955711,
    "climber": 10.233627,
    "coinrun": 9.93251,
    "dodgeball": 13.486584,
    "fruitbot": 29.925259,
    "heist": 9.685265,
    "jumper": 8.460201,
    "leaper": 7.4082565,
    "maze": 9.969294,
    "miner": 11.892558,
    "ninja": 9.474582,
    "plunder": 11.460528,
    "starpilot": 66.98625,
}
