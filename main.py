 
import argparse
import sys
from pathlib import Path
 
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
from RL.utils import *
from RL.policy import random_policy
from utils.exp import build_parser


# CONFIG = dict(
#     domain_name  = "walker",   # e.g. "walker", "cheetah", "hopper", "humanoid"
#     task_name    = "walk",     # depends on domain (see --list for options)
#     random_seed  = 42,
#     num_steps    = 300,        # simulation steps to record
#     camera_id    = 0,          # camera index used for rendering
#     render_width = 480,
#     render_height= 480,
#     output_dir   = Path("./figures"),  # where to save output files
# )

CONFIG = dict(
    env_id       = "Walker2d-v4",  # any Gymnasium MuJoCo env ID (see --list)
    random_seed  = 42,
    num_steps    = 500,            # max steps per episode to record
    render_width = 480,
    render_height= 480,
    camera_name  = None,           # None = default camera; or e.g. "track"
    fps          = 30,
    output_dir   = Path("./figures"),
)

def run(cfg: dict) -> None:
    env_id = cfg["env_id"]
 
    print(f"\n{'━' * 55}")
    print(f"  Gymnasium / MuJoCo  |  {env_id}")
    print(f"{'━' * 55}")
 
    # ── Build render kwargs ───────────────────────────────────────────────────
    # Gymnasium passes width/height/camera_name through the env constructor.
    render_kwargs = dict(
        width       = cfg["render_width"],
        height      = cfg["render_height"],
    )
    if cfg.get("camera_name"):
        render_kwargs["camera_name"] = cfg["camera_name"]
 
    # ── Load environment ──────────────────────────────────────────────────────
    # render_mode="rgb_array" makes env.render() return a numpy array.
    try:
        env = gym.make(
            env_id,
            render_mode = "rgb_array",
            **render_kwargs,
        )
    except TypeError:
        # Some older envs don't accept width/height in the constructor.
        env = gym.make(env_id, render_mode="rgb_array")
 
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
    frames  = []
    rewards = []
 
    obs, info = env.reset(seed=cfg["random_seed"])
    frames.append(env.render())   # capture the initial frame
 
    print(f"\n  Running up to {cfg['num_steps']} steps …")
    for step in range(cfg["num_steps"]):
        action = random_policy(obs, action_space)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
 
        # Render every other step to keep memory manageable
        if step % 2 == 0:
            frames.append(env.render())
 
        if terminated or truncated:
            print(f"    Episode ended at step {step + 1}  "
                  f"({'terminated' if terminated else 'truncated'})")
            break
 
    env.close()
 
    print(f"  Steps run    : {step + 1}")
    print(f"  Total reward : {sum(rewards):.2f}  "
          f"(mean {np.mean(rewards):.3f})")
 
    # ── Save outputs ──────────────────────────────────────────────────────────
    out  = Path(cfg["output_dir"])
    stem = env_id.replace("-", "_")
 
    save_snapshot_grid(frames, out / f"{stem}_snapshots.png")
    save_video(frames, out / f"{stem}.mp4", fps=cfg["fps"])
 
    print(f"\n  Done! Output files in: {out.resolve()}\n")
 
def main() -> None:
    args = build_parser(CONFIG).parse_args()
 
    # if args.list:
    #     list_environments()
    #     return
 
    cfg = {
        **CONFIG,
        "env_id"      : args.env,
        "num_steps"   : args.steps,
        "random_seed" : args.seed,
        "render_width": args.width,
        "render_height": args.height,
        "camera_name" : args.camera,
        "fps"         : args.fps,
        "output_dir"  : Path(args.out),
    }
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    run(cfg)

# def run(cfg: dict) -> None:
#     domain = cfg["domain_name"]
#     task   = cfg["task_name"]

#     print(f"\n{'━'*55}")
#     print(f"  dm_control  |  {domain} / {task}")
#     print(f"{'━'*55}")

#     # ── Load environment ──────────────────────────────────────────────────────
#     env = suite.load(
#         domain_name = domain,
#         task_name   = task,
#         task_kwargs = {"random": cfg["random_seed"]},
#     )

#     action_spec  = env.action_spec()
#     obs_spec     = env.observation_spec()

#     print(f"  Action space : {action_spec.shape}  "
#           f"[{action_spec.minimum.min():.2f}, {action_spec.maximum.max():.2f}]")
#     print(f"  Obs keys     : {list(obs_spec.keys())}")

#     # ── Simulation loop ───────────────────────────────────────────────────────
#     frames      = []
#     rewards     = []
#     time_step   = env.reset()

#     # Capture t=0 frame before any action
#     frames.append(render_frame(env, cfg))

#     print(f"\n  Running {cfg['num_steps']} steps …")
#     for step in range(cfg["num_steps"]):
#         action    = random_policy(time_step, action_spec)
#         time_step = env.step(action)
#         rewards.append(time_step.reward or 0.0)

#         # Render every other step to keep memory manageable
#         if step % 2 == 0:
#             frames.append(render_frame(env, cfg))

#         if time_step.last():
#             print(f"    Episode ended early at step {step}")
#             break

#     print(f"  Steps run    : {step + 1}")
#     print(f"  Total reward : {sum(rewards):.2f}  "
#           f"(mean {np.mean(rewards):.3f})")

#     # ── Save outputs ──────────────────────────────────────────────────────────
#     out  = Path(cfg["output_dir"])
#     stem = f"{domain}_{task}"

#     save_snapshot_grid(frames, out / f"{stem}_snapshots.png")
#     save_video(frames, out / f"{stem}.mp4")

#     print(f"\n  Done! Output files in: {out.resolve()}\n")

# def main() -> None:
#     args = build_parser(CONFIG).parse_args()


#     cfg = {
#         **CONFIG,
#         "domain_name"  : args.domain,
#         "task_name"    : args.task,
#         "num_steps"    : args.steps,
#         "random_seed"  : args.seed,
#         "render_width" : args.width,
#         "render_height": args.height,
#         "camera_id"    : args.camera,
#         "output_dir"   : Path(args.out),
#     }
#     cfg["output_dir"].mkdir(parents=True, exist_ok=True)
#     run(cfg)


if __name__ == "__main__":
    main()
