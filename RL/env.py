
import argparse
import os
import sys
from pathlib import Path

# ── Optional pretty output ────────────────────────────────────────────────────
try:
    from rich import print as rprint
    from rich.table import Table
    from rich.console import Console
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False

# ── dm_control / MuJoCo ───────────────────────────────────────────────────────
try:
    from dm_control import suite
    from dm_control.suite.wrappers import pixels
except ImportError:
    sys.exit(
        "dm_control not found. Install with:\n"
        "    pip install dm_control mujoco"
    )

import numpy as np


ALL_ENVIRONMENTS = {
    # ── Locomotion ────────────────────────────────────────────────────────────
    "walker":       ["stand", "walk", "run"],
    "hopper":       ["stand", "hop"],
    "cheetah":      ["run"],
    "humanoid":     ["stand", "walk", "run", "run_pure_state"],
    "humanoid_CMU": ["stand", "run"],

    # ── Classic control ───────────────────────────────────────────────────────
    "cartpole":     ["balance", "balance_sparse", "swingup", "swingup_sparse", "two_poles"],
    "acrobot":      ["swingup", "swingup_sparse"],
    "pendulum":     ["swingup"],
    "ball_in_cup":  ["catch"],
    "finger":       ["spin", "turn_easy", "turn_hard"],
    "reacher":      ["easy", "hard"],

    # ── Manipulation ──────────────────────────────────────────────────────────
    "manipulator":  ["bring_ball", "bring_peg", "insert_ball", "insert_peg"],
    "stacker":      ["stack_2", "stack_4"],

    # ── Swimming / aquatic ────────────────────────────────────────────────────
    "fish":         ["upright", "swim"],
    "swimmer":      ["swimmer6", "swimmer15"],

    # ── Point-mass ────────────────────────────────────────────────────────────
    "point_mass":   ["easy", "hard"],

    # ── Loco-manipulation ─────────────────────────────────────────────────────
    "dog":          ["stand", "walk", "trot", "run", "fetch", "fetch_harder"],
    "quadruped":    ["walk", "run", "escape", "fetch"],
}


def list_environments() -> None:
    """Pretty-print every domain / task pair."""
    if USE_RICH:
        table = Table(title="dm_control Suite – Available Environments",
                      show_lines=True)
        table.add_column("Domain", style="cyan bold", no_wrap=True)
        table.add_column("Tasks", style="green")
        table.add_column("Category", style="yellow")

        categories = {
            "Locomotion":        ["walker","hopper","cheetah","humanoid","humanoid_CMU"],
            "Classic control":   ["cartpole","acrobot","pendulum","ball_in_cup",
                                   "finger","reacher"],
            "Manipulation":      ["manipulator","stacker"],
            "Swimming/aquatic":  ["fish","swimmer"],
            "Point-mass":        ["point_mass"],
            "Loco-manipulation": ["dog","quadruped"],
        }
        domain_to_cat = {d: c for c, ds in categories.items() for d in ds}

        for domain, tasks in ALL_ENVIRONMENTS.items():
            table.add_row(domain, ", ".join(tasks),
                          domain_to_cat.get(domain, ""))
        console.print(table)
    else:
        print(f"\n{'Domain':<20} {'Tasks'}")
        print("─" * 70)
        for domain, tasks in ALL_ENVIRONMENTS.items():
            print(f"  {domain:<18} {', '.join(tasks)}")
        print()
