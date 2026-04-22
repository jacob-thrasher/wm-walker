
from shimmy.dm_control_compatibility import DmControlCompatibilityV0
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)


from RL.env import load_environment


render_kwargs = dict(
        width       = 64,
        height      = 64,
    )
# render_kwargs["camera_name"] = cfg["camera_name"]
env = load_environment('Walker2d-v4', render_kwargs)


model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    device='cuda'
)

model.learn(total_timesteps=1_000_000)

model.save('models/ppo')