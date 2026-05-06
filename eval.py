import torch
import os
import config
import env_utils
import paths
import matplotlib.pyplot as plt
from tqdm import tqdm
from env_utils import RealtimeViewer
from ppo import create_buffer
from data_loader import normalize_obs
from wm import EvalWorldModel
import data_loader

device = 'cuda'
doViz = True

state_dict_policy = torch.load(paths.get_decoded_policy_path(config.get().exp_name), weights_only=False)
state_dict_idm = torch.load(paths.get_models_path(config.get().exp_name), weights_only=False)
cfg = config.get(base_cfg=state_dict_policy["cfg"], reload_keys=["stage2", "stage3"])
wm = EvalWorldModel(state_dict_policy, state_dict_idm)


train_data, valid_data = data_loader.load('walker')
train_dataloader = train_data.get_iter(64)
valid_dataloader = valid_data.get_iter(64)

batch = next(iter(valid_dataloader))
obs_in_seq = batch['obs'].to(device)
action = batch['ta'].to(device)
out, mu, logvar = wm.idm(obs_in_seq)
print(out['la'])
print(mu)
print()
print(logvar)
input()


envs = env_utils.setup_gym_env_vectorized(env_id='Walker2d-v4', 
                                          num_envs=cfg.stage3.num_envs, 
                                          render_kwargs=dict(
                                                            width       = 64,
                                                            height      = 64,
                                                        )
                                        )

# envs = env_utils.build_vec_env(env_id='Walker2d-v4', n_envs=1, seed=69, use_subproc=False)

rl_cfg = cfg.stage3
obs = envs.reset()
obs = torch.tensor(envs.render())
next_obs = obs.permute((0, 3, 1, 2)).to(device)
last_obs = None
next_done = torch.zeros(rl_cfg.num_envs).to(device).float()
print(f'Executing {rl_cfg.num_steps} steps in {rl_cfg.num_envs} environments')

buf = create_buffer(
        rl_cfg.num_steps,
        rl_cfg.num_envs,
        envs.single_observation_space,
        envs.single_action_space,
        device,
    )

for step in range(0, 3):
    next_obs = normalize_obs(next_obs)
    buf['obs'][step] = next_obs
    buf['dones'][step] = next_done
    action, logprobs, _, value, (latent_action, fdm_out) = wm.action_selection_hook(next_obs)
    buf["values"][step] = value.flatten()
    print(latent_action)
    last_obs = next_obs
    _ = envs.step(action.cpu().numpy())
    next_obs = torch.tensor(envs.render()).permute((0, 3, 1, 2)).to(device)


if doViz:
    viz = RealtimeViewer()
    viz.initialize(next_obs, method='wide')


# ---- Execution ----------
rewards = []
for step in tqdm(range(0, rl_cfg.num_steps), desc=f'Executing {rl_cfg.num_envs} envs', disable=False):
    buf['obs'][step] = next_obs
    buf['dones'][step] = next_done
    display = next_obs[0]
    next_obs = normalize_obs(next_obs)


    action, logprobs, _, value, (latent_action, fdm_out) = wm.action_selection_hook(next_obs)


    if len(wm.buf_obs) == 3 and doViz:
        frames = wm.imagine()
        viz.update([wm.buf_obs[0][0], wm.buf_obs[1][0], wm.buf_obs[2][0],
                    frames[0][0].squeeze(), frames[1][0].squeeze(), frames[2][0].squeeze()])



    # Advance environment
    buf["values"][step] = value.flatten()
    last_obs = next_obs
    _ = envs.step(action.cpu().numpy())
    next_obs = torch.tensor(envs.render()).permute((0, 3, 1, 2)).to(device)
    # rewards.append(reward)