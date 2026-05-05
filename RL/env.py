
# import argparse
# import os
# import sys
# from pathlib import Path

# # ── Optional pretty output ────────────────────────────────────────────────────
# try:
#     from rich import print as rprint
#     from rich.table import Table
#     from rich.console import Console
#     console = Console()
#     USE_RICH = True
# except ImportError:
#     USE_RICH = False

# # ── dm_control / MuJoCo ───────────────────────────────────────────────────────
# try:
#     from dm_control import suite
#     from dm_control.suite.wrappers import pixels
# except ImportError:
#     sys.exit(
#         "dm_control not found. Install with:\n"
#         "    pip install dm_control mujoco"
#     )

# import numpy as np


# ALL_ENVIRONMENTS = {
#     # ── Locomotion ────────────────────────────────────────────────────────────
#     "walker":       ["stand", "walk", "run"],
#     "hopper":       ["stand", "hop"],
#     "cheetah":      ["run"],
#     "humanoid":     ["stand", "walk", "run", "run_pure_state"],
#     "humanoid_CMU": ["stand", "run"],

#     # ── Classic control ───────────────────────────────────────────────────────
#     "cartpole":     ["balance", "balance_sparse", "swingup", "swingup_sparse", "two_poles"],
#     "acrobot":      ["swingup", "swingup_sparse"],
#     "pendulum":     ["swingup"],
#     "ball_in_cup":  ["catch"],
#     "finger":       ["spin", "turn_easy", "turn_hard"],
#     "reacher":      ["easy", "hard"],

#     # ── Manipulation ──────────────────────────────────────────────────────────
#     "manipulator":  ["bring_ball", "bring_peg", "insert_ball", "insert_peg"],
#     "stacker":      ["stack_2", "stack_4"],

#     # ── Swimming / aquatic ────────────────────────────────────────────────────
#     "fish":         ["upright", "swim"],
#     "swimmer":      ["swimmer6", "swimmer15"],

#     # ── Point-mass ────────────────────────────────────────────────────────────
#     "point_mass":   ["easy", "hard"],

#     # ── Loco-manipulation ─────────────────────────────────────────────────────
#     "dog":          ["stand", "walk", "trot", "run", "fetch", "fetch_harder"],
#     "quadruped":    ["walk", "run", "escape", "fetch"],
# }


# def list_environments() -> None:
#     """Pretty-print every domain / task pair."""
#     if USE_RICH:
#         table = Table(title="dm_control Suite – Available Environments",
#                       show_lines=True)
#         table.add_column("Domain", style="cyan bold", no_wrap=True)
#         table.add_column("Tasks", style="green")
#         table.add_column("Category", style="yellow")

#         categories = {
#             "Locomotion":        ["walker","hopper","cheetah","humanoid","humanoid_CMU"],
#             "Classic control":   ["cartpole","acrobot","pendulum","ball_in_cup",
#                                    "finger","reacher"],
#             "Manipulation":      ["manipulator","stacker"],
#             "Swimming/aquatic":  ["fish","swimmer"],
#             "Point-mass":        ["point_mass"],
#             "Loco-manipulation": ["dog","quadruped"],
#         }
#         domain_to_cat = {d: c for c, ds in categories.items() for d in ds}

#         for domain, tasks in ALL_ENVIRONMENTS.items():
#             table.add_row(domain, ", ".join(tasks),
#                           domain_to_cat.get(domain, ""))
#         console.print(table)
#     else:
#         print(f"\n{'Domain':<20} {'Tasks'}")
#         print("─" * 70)
#         for domain, tasks in ALL_ENVIRONMENTS.items():
#             print(f"  {domain:<18} {', '.join(tasks)}")
#         print()


# def load_environment(domain='walker', task='walk', seed=42):
#     env = suite.load(
#         domain_name = domain,
#         task_name   = task,
#         task_kwargs = {"random": seed},
#     )
#     return env

import argparse
import sys
from pathlib import Path
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
 
# ── Optional pretty output ────────────────────────────────────────────────────
try:
    from rich.table import Table
    from rich.console import Console
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
 
# ── NumPy ─────────────────────────────────────────────────────────────────────
import numpy as np
 
# ── Gymnasium ─────────────────────────────────────────────────────────────────
try:
    import gymnasium as gym
except ImportError:
    sys.exit(
        "\n  ✗  gymnasium not found. Install with:\n\n"
        "         pip install gymnasium[mujoco] mujoco\n"
    )
 

ALL_ENVIRONMENTS = {
    # ── Locomotion ─────────────────────────────────────────────────────────
    "Walker2d-v4":              "Bipedal walker — 2-D sagittal plane",
    "Walker2d-v5":              "Walker2d (v5 API)",
    "HalfCheetah-v4":           "Planar cheetah — fastest locomotion env",
    "HalfCheetah-v5":           "HalfCheetah (v5 API)",
    "Hopper-v4":                "One-legged hopper — balance + forward speed",
    "Hopper-v5":                "Hopper (v5 API)",
    "Humanoid-v4":              "3-D humanoid — 17 DoF, hardest locomotion",
    "Humanoid-v5":              "Humanoid (v5 API)",
    "HumanoidStandup-v4":       "Humanoid — reward for rising from floor",
    "HumanoidStandup-v5":       "HumanoidStandup (v5 API)",
    "Ant-v4":                   "Quadruped ant — 3-D locomotion",
    "Ant-v5":                   "Ant (v5 API)",
    "Swimmer-v4":               "3-link swimmer in viscous fluid",
    "Swimmer-v5":               "Swimmer (v5 API)",
 
    # ── Classic control ────────────────────────────────────────────────────
    "InvertedPendulum-v4":      "Cartpole-style balance task",
    "InvertedPendulum-v5":      "InvertedPendulum (v5 API)",
    "InvertedDoublePendulum-v4":"Double pendulum balance — very sensitive",
    "InvertedDoublePendulum-v5":"InvertedDoublePendulum (v5 API)",
 
    # ── Manipulation ───────────────────────────────────────────────────────
    "Reacher-v4":               "2-DoF arm — reach a random target",
    "Reacher-v5":               "Reacher (v5 API)",
    "Pusher-v4":                "7-DoF arm — push object to goal",
    "Pusher-v5":                "Pusher (v5 API)",
}
 
 
def list_environments() -> None:
    """Pretty-print every Gymnasium MuJoCo environment."""
    if USE_RICH:
        table = Table(
            title="Gymnasium MuJoCo – Available Environments",
            show_lines=True,
        )
        table.add_column("Env ID",      style="cyan bold", no_wrap=True)
        table.add_column("Description", style="green")
        for env_id, desc in ALL_ENVIRONMENTS.items():
            table.add_row(env_id, desc)
        console.print(table)
    else:
        print(f"\n  {'Env ID':<35} Description")
        print("  " + "─" * 70)
        for env_id, desc in ALL_ENVIRONMENTS.items():
            print(f"  {env_id:<35} {desc}")
        print()

def load_environment(env_id, render_kwargs, num_envs=1, ):
    env = gym.make(
            env_id,
            # num_envs=num_envs,
            render_mode = "rgb_array",
            **render_kwargs,
        )
    return env

def load_vec_environment(env_id, vec_path="./logs/walker2d/vecnormalize_final.pkl", render_kwargs=None):

    env = gym.make(
            env_id,
            render_mode = "rgb_array",
            **render_kwargs,
        )

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(
        vec_path,
        vec_env,
    )
    vec_env.training = False
    vec_env.norm_reward = False

    return vec_env