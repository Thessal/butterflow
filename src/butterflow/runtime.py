from butterflow.operators import Node


class Runtime:
    def __init__(self, data=None):
        self.cache = {}
        if data:
            for k,v in data.items():
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
            return node.compute(cache = cache)
        else:
            raise Exception("Root node is not a signal")
