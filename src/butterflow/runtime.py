from butterflow.operators import Node
from glob import glob
import numpy as np


def filename_to_id(path):
    return path.split("/")[-1].split(".")[0]


def id_to_expr(x):
    return f'data("{x}")'


def check_expr(x):
    return x.startswith('data("') and x.endswith('")')


class Runtime:
    def __init__(self, data=None, datadir=None):
        self.cache = {}
        if datadir:  # load all in datadir
            files = glob(f"{datadir}/*.npy")
            data = {filename_to_id(file): np.load(file) for file in files}
            if not ("returns" in data):
                x_close = data["close"]
                x_close_d1 = np.roll(x_close, shift=1, axis=0)
                x_close_d1[0] = x_close[0]
                x_ret = (x_close / x_close_d1) - 1
                data["returns"] = x_ret
        if data:  # manually set set up cache
            for k, v in data.items():
                if not check_expr(k):
                    k = id_to_expr(k)
                self.cache[k] = v

    def run(self, node, cache=None):
        """
        Recursively evaluates a Node and its dependencies (Post-Order Traversal).
        """
        # Memoization
        if cache is None:
            cache = self.cache

        # TODO: data op cache

        # Root node is operator
        if issubclass(type(node), Node):
            return node.compute(cache=cache)
        else:
            raise Exception("Root node is not a signal")
