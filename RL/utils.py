from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

def render_frame(env, cfg: dict) -> np.ndarray:
    return env.physics.render(
        height    = cfg["render_height"],
        width     = cfg["render_width"],
        camera_id = cfg["camera_id"],
    )


def save_snapshot_grid(frames: list, path: Path, n_cols: int = 5) -> None:
    """Save an evenly-spaced grid of frames as a PNG."""

    indices = np.linspace(0, len(frames) - 1, min(n_cols * 2, len(frames)),
                          dtype=int)
    selected = [frames[i] for i in indices]
    n_cols   = min(n_cols, len(selected))
    n_rows   = (len(selected) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 3))
    axes = np.array(axes).flatten()

    for ax, frame in zip(axes, selected):
        ax.imshow(frame)
        ax.axis("off")
    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(f"{path.stem}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Snapshot grid  → {path}")


def save_video(frames: list, path: Path, fps: int = 30) -> None:
    """Save frames as an MP4 video using imageio (v2 and v3 compatible)."""

    # Drop any None frames that can appear at episode boundaries
    clean = [f for f in frames if f is not None]
    if not clean:
        print("  ⚠  No valid frames to write – skipping video.")
        return

    frames_u8 = [f.astype(np.uint8) for f in clean]

    try:
        # Works for both imageio v2 and v3 — use get_writer for full control
        with imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            # output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        ) as writer:
            for frame in frames_u8:
                writer.append_data(frame)
        print(f"  ✓  Video           → {path}")
    except Exception as exc_primary:
        # Bare fallback: no extra kwargs at all
        try:
            imageio.mimwrite(str(path), frames_u8, fps=fps)
            print(f"  ✓  Video           → {path}")
        except Exception as exc_fallback:
            print(
                f"  ⚠  Video export failed.\n"
                f"     Primary error  : {exc_primary}\n"
                f"     Fallback error : {exc_fallback}\n"
                f"     Try: pip install imageio[ffmpeg]"
            )

