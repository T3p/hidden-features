from .datasets.jester import Jester, Jester_fitted
import os

def load(name, data_path, random_state=0):
    env = None
    if name == "jester_v0":
        env = Jester(
            data_file=os.path.join(data_path, "jester/jester_svd36.npz"),
            random_state=random_state
        )
    elif name == "jester_lin":
        env = Jester_fitted(
            data_file=os.path.join(data_path, "jester/jester_linear_fitted.npz"), 
            random_state=random_state
        )
    else:
        raise ValueError(f"!unregistered problem ({name})")
    return env