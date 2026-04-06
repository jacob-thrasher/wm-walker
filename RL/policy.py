import numpy as np

def random_policy(time_step, action_spec):
    """Uniformly random actions – a useful baseline / sanity check."""
    np.random.seed(int(time_step.step_type) + 1)
    return np.random.uniform(
        low  = action_spec.minimum,
        high = action_spec.maximum,
        size = action_spec.shape,
    )


def zero_policy(time_step, action_spec):
    """All-zeros – useful for testing passive dynamics."""
    return np.zeros(action_spec.shape)
