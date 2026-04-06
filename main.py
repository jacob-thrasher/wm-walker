from pathlib import Path
from dm_control import suite
from dm_control.suite.wrappers import pixels

from RL.utils import *
from RL.policy import random_policy
from utils.exp import build_parser


CONFIG = dict(
    domain_name  = "walker",   # e.g. "walker", "cheetah", "hopper", "humanoid"
    task_name    = "walk",     # depends on domain (see --list for options)
    random_seed  = 42,
    num_steps    = 300,        # simulation steps to record
    camera_id    = 0,          # camera index used for rendering
    render_width = 480,
    render_height= 480,
    output_dir   = Path("./figures"),  # where to save output files
)

def run(cfg: dict) -> None:
    domain = cfg["domain_name"]
    task   = cfg["task_name"]

    print(f"\n{'━'*55}")
    print(f"  dm_control  |  {domain} / {task}")
    print(f"{'━'*55}")

    # ── Load environment ──────────────────────────────────────────────────────
    env = suite.load(
        domain_name = domain,
        task_name   = task,
        task_kwargs = {"random": cfg["random_seed"]},
    )

    action_spec  = env.action_spec()
    obs_spec     = env.observation_spec()

    print(f"  Action space : {action_spec.shape}  "
          f"[{action_spec.minimum.min():.2f}, {action_spec.maximum.max():.2f}]")
    print(f"  Obs keys     : {list(obs_spec.keys())}")

    # ── Simulation loop ───────────────────────────────────────────────────────
    frames      = []
    rewards     = []
    time_step   = env.reset()

    # Capture t=0 frame before any action
    frames.append(render_frame(env, cfg))

    print(f"\n  Running {cfg['num_steps']} steps …")
    for step in range(cfg["num_steps"]):
        action    = random_policy(time_step, action_spec)
        time_step = env.step(action)
        rewards.append(time_step.reward or 0.0)

        # Render every other step to keep memory manageable
        if step % 2 == 0:
            frames.append(render_frame(env, cfg))

        if time_step.last():
            print(f"    Episode ended early at step {step}")
            break

    print(f"  Steps run    : {step + 1}")
    print(f"  Total reward : {sum(rewards):.2f}  "
          f"(mean {np.mean(rewards):.3f})")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out  = Path(cfg["output_dir"])
    stem = f"{domain}_{task}"

    save_snapshot_grid(frames, out / f"{stem}_snapshots.png")
    save_video(frames, out / f"{stem}.mp4")

    print(f"\n  Done! Output files in: {out.resolve()}\n")

def main() -> None:
    args = build_parser(CONFIG).parse_args()


    cfg = {
        **CONFIG,
        "domain_name"  : args.domain,
        "task_name"    : args.task,
        "num_steps"    : args.steps,
        "random_seed"  : args.seed,
        "render_width" : args.width,
        "render_height": args.height,
        "camera_id"    : args.camera,
        "output_dir"   : Path(args.out),
    }
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    run(cfg)


if __name__ == "__main__":
    main()
