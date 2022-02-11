from dataclasses import dataclass
import numpy as np

@dataclass
class DiscreteFix:
    n: int

    def contains(self, x) -> bool:
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)  # type: ignore
        else:
            return False
        return 0 <= as_int < self.n