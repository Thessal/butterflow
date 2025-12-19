from .typesystem import Operator, Atomic, Generic, DictType, Either
import numpy as np
import scipy

# --- Define the Type Environment (The Rules) ---
# Function signatures
# Mapping function names to their Operator signatures based on your input
STD_LIB = {
    "data": Operator(
        DictType({
            "id": Atomic("String")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_mean": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Either([Generic("Signal", Atomic("Float")), Atomic("Int")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "divide": Operator(
        DictType({
            "dividend": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "divisor":  Either([Generic("Signal", Atomic("Float")), Atomic("Float")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "multiply": Operator(
        DictType({
            "baseline": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "multiplier":  Either([Generic("Signal", Atomic("Float")), Atomic("Float")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "covariance": Operator(
        DictType({
            "returns": Generic("Signal", Atomic("Float")),
            "lookback": Atomic("Int")
        }),
        Generic("Matrix", Atomic("Float"))
    ),

}


class Node:
    def __repr__(self):
        args = ", ".join(
            f"{v}" for k, v in self.__dict__.items() if not k.startswith('_'))
        return f"{self.__class__.__name__}({args})"

    def get_kwargs(self):
        op_name = type(self).__name__
        kws = STD_LIB[op_name].args.fields.keys()
        kwargs = {kw: getattr(self, kw) for kw in kws}
        return kwargs

    def compute(self, cache):
        # Check cache
        # cache_keys = repr(list(cache.keys()))
        # print("cache_keys", cache_keys)
        expr_str = repr(self)
        if cache and (expr_str in cache):
            return cache[expr_str]

        # Compute
        kwargs = dict()
        # print(f"calculating {self.__class__.__name__}({self.get_kwargs()})")
        for kw, arg in self.get_kwargs().items():
            if issubclass(type(arg), Node):
                kwargs[kw] = arg.compute(cache)
            else:
                kwargs[kw] = arg
            # print(f"Arg : {kw} = {type(kwargs[kw])}")
        # kwargs = {k: v.compute() for k, v in self.get_args()}
        output = self._compute(**kwargs)
        if cache:
            cache[expr_str] = output

        return output


class STD_LIB_IMPL:
    @staticmethod
    def get(name):
        # Get subclass from string
        return getattr(STD_LIB_IMPL, name)

    # Graph Nodes
    class data(Node):
        def __init__(self, id): self.id = id
        def __repr__(self): return f'data("{self.id}")'

        def _compute(self, id):
            raise Exception(f"Data operator should be fetched from cache, and cannot be directly calculated.\nMissing item: {repr(self)}")
            # return np.ones(shape=(10, 10))

    class ts_mean(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            n, p = signal.shape
            result = np.full_like(signal, np.nan, dtype=float)
            if type(period) == int:
                if 0 < period <= n : 
                    kernel = np.ones((period,1), dtype=float) * (1./float(period))
                    result[period-1:] = scipy.signal.convolve2d(signal, kernel, mode='valid')
                return result
            elif type(period) == np.ndarray:
                assert signal.shape == period.shape
                lookback = np.round(period).clip(min=0, max=n+1)
                lookback = np.nan_to_num(lookback, nan=0, posinf=0, neginf=0).astype(int)
                lbs = [int(x) for x in np.unique(lookback)]
                for lb in lbs :
                    mask = lookback == lb
                    result[mask] = self._compute(signal, lb)[mask]
                return result
            else:
                raise Exception

    class divide(Node):
        def __init__(self, dividend,
                     divisor): self.dividend, self.divisor = dividend, divisor
        def _compute(self, dividend,
                     divisor):
            return dividend / divisor

    class multiply(Node):
        def __init__(self, baseline,
                     multiplier):
            self.baseline, self.multiplier = baseline, multiplier

        def _compute(self, baseline, multiplier):
            return baseline * multiplier
    
    class covariance(Node):
        # TODO: shrinkage or regularization, neutralization, sector information etc.
        def __init__(self, returns, lookback):
            self.returns, self.lookback = returns, lookback

        def _compute(self, returns, lookback):
            n, p = returns.shape
            result = np.full((n,p,p), np.nan, dtype=float)
            for t in range(lookback,n):
                result[t] = np.cov(returns[t-lookback: t+1], rowvar=False)
            return result