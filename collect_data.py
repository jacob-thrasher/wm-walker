import numpy as np

from RL.env import load_environment
from RL.policy import random_policy
from RL.utils import save_video

cfg = dict(
    env_id       = "Walker2d-v4",  # any Gymnasium MuJoCo env ID (see --list)
    random_seed  = 42,
    num_steps    = 500,            # max steps per episode to record
    render_width = 224,
    render_height= 224,
    camera_name  = None,           # None = default camera; or e.g. "track"
    fps          = 30,
    output_dir   = './data'
)

render_kwargs = dict(
        width       = 224,
        height      = 224,
    )
# render_kwargs["camera_name"] = cfg["camera_name"]
env = load_environment('Walker2d-v4', render_kwargs)
obs_space    = env.observation_space
action_space = env.action_space

print(f"  Obs space    : shape={obs_space.shape}  "
        f"[{obs_space.low.min():.2f}, {obs_space.high.max():.2f}]")
print(f"  Action space : shape={action_space.shape}  "
        f"[{action_space.low.min():.2f}, {action_space.high.max():.2f}]")

# ── Simulation loop ───────────────────────────────────────────────────────
# Gymnasium API (>= 0.26):
#   reset() → (obs, info)
#   step()  → (obs, reward, terminated, truncated, info)
observations = []
true_actions = []
dones = []
rewards = []
frames  = []

obs, info = env.reset(seed=cfg["random_seed"])
frames.append(env.render())   # capture the initial frame

print(f"\n  Running up to {cfg['num_steps']} steps …")
for step in range(cfg["num_steps"]):
    action = random_policy(obs, action_space)
    obs, reward, terminated, truncated, info = env.step(action)
    # rewards.append(reward)
    isDone = terminated or truncated

    
    # Render every other step to keep memory manageable
    if step % 2 == 0:
        frames.append(env.render())
        true_actions.append(action)
        dones.append(isDone)
        rewards.append(reward)

    if isDone:
        print(f"    Episode ended at step {step + 1}  "
                f"({'terminated' if terminated else 'truncated'})")

        obs, info = env.reset()

env.close()


np.savez('./data/test.npz', obs=frames, ta=true_actions, dones=dones, rewards=rewards)

# ── Save outputs ─────────────────────────────────────────────────
stem = 'test'

save_video(frames, f"./figures/{stem}.mp4", fps=cfg["fps"])

 



