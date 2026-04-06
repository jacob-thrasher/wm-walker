import argparse

def build_parser(config) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description = "dm_control walker demo – flexible environment runner",
        formatter_class = argparse.RawTextHelpFormatter,
    )
    p.add_argument("--domain", default=config["domain_name"],
                   help="dm_control domain  (default: %(default)s)")
    p.add_argument("--task",   default=config["task_name"],
                   help="Task within domain (default: %(default)s)")
    p.add_argument("--steps",  type=int, default=config["num_steps"],
                   help="Simulation steps   (default: %(default)s)")
    p.add_argument("--seed",   type=int, default=config["random_seed"],
                   help="Random seed        (default: %(default)s)")
    p.add_argument("--width",  type=int, default=config["render_width"])
    p.add_argument("--height", type=int, default=config["render_height"])
    p.add_argument("--camera", type=int, default=config["camera_id"],
                   help="MuJoCo camera id   (default: %(default)s)")
    p.add_argument("--out",    default=str(config["output_dir"]),
                   help="Output directory   (default: %(default)s)")
    p.add_argument("--list",   action="store_true",
                   help="Print all available environments and exit")
    return p
