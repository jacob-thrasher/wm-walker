import numpy as np
import sys
from pathlib import Path
from stable_baselines3 import PPO, SAC
from tqdm import tqdm

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from RL.env import load_environment, load_vec_environment
from RL.policy import random_policy
from RL.utils import save_video


# TODO: Refactor to collect large amounts of data
def run_model(cfg, render_kwargs, out_dir, max_iter=100):
    # env = load_environment('Walker2d-v4', render_kwargs=render_kwargs)
    env = load_vec_environment('Walker2d-v4', render_kwargs=render_kwargs)
    obs_space    = env.observation_space
    action_space = env.action_space

    model = SAC.load('./logs/walker2d/best_model/best_model.zip', env=env)


    # ── Simulation loop ───────────────────────────────────────────────────────

    observations = []
    true_actions = []
    dones = []
    rewards = []
    frames  = []

    obs = env.reset()
    # frames.append(env.render())   # capture the initial frame
    counter = 0
    print(f"\n  Running up to {cfg['num_steps']} steps …")
    for step in range(cfg["num_steps"]):
        # action = random_policy(obs, action_space)
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, done, info = env.step(action)
        # rewards.append(reward)
        
        # Render every other step to keep memory manageable
        # if step % 2 == 0:
        frames.append(env.render())
        true_actions.append(action)
        dones.append(done)
        rewards.append(reward)

        if done:
            obs = env.reset()

        # if counter > max_iter: break

    env.close()

    np.savez(out_dir, 
             obs=frames, 
             ta=true_actions, 
             done=dones, 
             rewards=rewards, 
             ep_returns=[0]*len(frames), 
             values=[0]*len(frames))




TRAIN_CHUNK_LEN = 8128
TEST_CHUNK_LEN = 4096

for i in tqdm(range(20), desc='Gathering train data'):
    cfg = dict(
        env_id       = "Walker2d-v4",  # any Gymnasium MuJoCo env ID (see --list)
        random_seed  = 42,
        num_steps    = TRAIN_CHUNK_LEN,            # max steps per episode to record
        render_width = 64,
        render_height= 64,
        camera_name  = None,           # None = default camera; or e.g. "track"
        fps          = 30,
        output_dir   = './data'
    )

    render_kwargs = dict(
            width       = 64,
            height      = 64,
        )

    run_model(cfg, render_kwargs, f'./expert_data/walker/train/{i}.npz')

for i in tqdm(range(20), desc='Gathering test data'):
    cfg = dict(
        env_id       = "Walker2d-v4",  # any Gymnasium MuJoCo env ID (see --list)
        random_seed  = 42,
        num_steps    = TEST_CHUNK_LEN,            # max steps per episode to record
        render_width = 64,
        render_height= 64,
        camera_name  = None,           # None = default camera; or e.g. "track"
        fps          = 30,
        output_dir   = './data'
    )

    render_kwargs = dict(
            width       = 64,
            height      = 64,
        )

    run_model(cfg, render_kwargs, f'./expert_data/walker/test/{i}.npz')


# save_video(frames, f"./figures/{stem}.mp4", fps=cfg["fps"])

 



