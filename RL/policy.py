import numpy as np
import gymnasium as gym

# def random_policy(time_step, action_spec):
#     """Uniformly random actions – a useful baseline / sanity check."""
#     np.random.seed(int(time_step.step_type) + 1)
#     return np.random.uniform(
#         low  = action_spec.minimum,
#         high = action_spec.maximum,
#         size = action_spec.shape,
#     )


# def zero_policy(time_step, action_spec):
#     """All-zeros – useful for testing passive dynamics."""
#     return np.zeros(action_spec.shape)


def random_policy(obs: np.ndarray, action_space: gym.Space) -> np.ndarray:
    """Uniformly random actions sampled from the action space."""
    return action_space.sample()
 
 
def zero_policy(obs: np.ndarray, action_space: gym.Space) -> np.ndarray:
    """All-zeros — useful for inspecting passive dynamics."""
    return np.zeros(action_space.shape, dtype=action_space.dtype)