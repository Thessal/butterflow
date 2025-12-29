from typesystem import Operator, Atomic, Generic, DictType, Either
import numpy as np
import scipy
from bisect import bisect_left, bisect_right, insort

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

    "const": Operator(
        DictType({
            "value": Atomic("Float")
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

    "ts_std": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_zscore": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_rank": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_ffill": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "cs_rank": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "cs_zscore": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
        }),
        Generic("Signal", Atomic("Float"))
    ),


    "add": Operator(
        DictType({
            "x": Generic("Signal", Atomic("Float")),
            "y": Generic("Signal", Atomic("Float")),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "subtract": Operator(
        DictType({
            "x": Generic("Signal", Atomic("Float")),
            "y": Generic("Signal", Atomic("Float")),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "divide": Operator(
        DictType({
            "dividend": Generic("Signal", Atomic("Float")),
            "divisor":  Generic("Signal", Atomic("Float"))
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "multiply": Operator(
        DictType({
            "baseline": Generic("Signal", Atomic("Float")),
            "multiplier": Either([Generic("Signal", Atomic("Float")), Atomic("Float")])
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
        self.cache = cache
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
            raise Exception(
                f"Data operator should be fetched from cache, and cannot be directly calculated.\nMissing item: {repr(self)}")
            # return np.ones(shape=(10, 10))

    class const(Node):
        def __init__(self, value): self.value = value
        def __repr__(self): return f'const("{self.value}")'

        def _compute(self, value):
            close = STD_LIB_IMPL.data(id="close").compute(cache=self.cache)
            close[:, :] = value
            return close

    class ts_mean(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            n, p = signal.shape
            result = np.full_like(signal, np.nan, dtype=float)
            if type(period) == int:
                if 0 < period <= n:
                    kernel = np.ones((period, 1), dtype=float) * \
                        (1./float(period))
                    result[period -
                           1:] = scipy.signal.convolve2d(signal, kernel, mode='valid')
                return result
            elif type(period) == np.ndarray:
                assert signal.shape == period.shape
                lookback = np.round(period).clip(min=0, max=n+1)
                lookback = np.nan_to_num(
                    lookback, nan=0, posinf=0, neginf=0).astype(int)
                lbs = [int(x) for x in np.unique(lookback)]
                for lb in lbs:
                    mask = lookback == lb
                    result[mask] = self._compute(signal, lb)[mask]
                return result
            else:
                raise Exception

    class ts_std(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            n, p = signal.shape
            result = np.full_like(signal, np.nan, dtype=float)
            if type(period) == int:
                if 0 < period <= n:
                    for p in range(period, n+1):
                        result[p-1] = signal[p-period:p].std(axis=0)
                return result
            else:
                raise Exception

    class ts_zscore(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            if type(period) == int:
                mu = STD_LIB_IMPL.ts_mean(
                    signal=signal, period=period).compute(self.cache)
                sigm = STD_LIB_IMPL.ts_std(
                    signal=signal, period=period).compute(self.cache)
                z = STD_LIB_IMPL.divide(
                    dividend=signal-mu, divisor=sigm).compute(self.cache)
                return z
            else:
                raise Exception

    class ts_rank(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal: np.ndarray, period: int) -> np.ndarray:
            """
            Parameters
            ----------
            signal : ndarray, shape (T, N)
                axis 0: rolling dimension (time)
                axis 1: independent series
            period : int, positive

            Returns
            -------
            out : ndarray, shape (T, N)
                Rolling rank of x[t, j] within x[t-period+1:t+1, j]
            """
            x = np.asarray(signal, dtype=float)
            T, N = x.shape
            out = np.full((T, N), np.nan)

            if period > 0:
                # one sorted buffer per column
                buffers = [[] for _ in range(N)]

                for t in range(T):
                    for j in range(N):
                        v = x[t, j]
                        buf = buffers[j]

                        # insert new value
                        if not np.isnan(v):
                            insort(buf, v)

                        # remove leaving value
                        if t >= period:
                            old = x[t - period, j]
                            if not np.isnan(old):
                                idx = bisect_left(buf, old)
                                del buf[idx]

                        # compute rank
                        if t >= period - 1 and not np.isnan(v):
                            left = bisect_left(buf, v)
                            right = bisect_right(buf, v)
                            out[t, j] = left + (right - left + 1) / 2

            return out

    class ts_ffill(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal: np.ndarray, period: int) -> np.ndarray:
            """
            NumPy implementation of pandas.DataFrame.ffill(limit=limit)

            Parameters
            ----------
            signal : np.ndarray (2D)
                Input array with NaNs.
            period : int
                Maximum number of consecutive NaNs to fill.

            Returns
            -------
            out : np.ndarray
                Forward-filled array.
            """
            arr = np.asarray(signal)
            if arr.ndim != 2:
                raise ValueError("Input must be 2D")

            n, m = arr.shape
            out = arr.copy()

            for j in range(m):
                col = out[:, j]

                valid = ~np.isnan(col)
                idx = np.where(valid, np.arange(n), -1)

                # last valid index up to each row
                last = np.maximum.accumulate(idx)

                # distance from last valid
                dist = np.arange(n) - last

                # fill condition
                fill_mask = (~valid) & (last >= 0) & (dist <= period)

                col[fill_mask] = col[last[fill_mask]]

            return out

    class cs_rank(Node):
        def __init__(self, signal): self.signal = signal

        def _compute(self, signal: np.ndarray) -> np.ndarray:
            """
            NumPy equivalent of pandas.DataFrame.rank(axis=1)
            - method='average'
            - ascending=True
            - na_option='keep'
            """
            x = np.asarray(signal, dtype=float)
            out = np.full_like(x, np.nan)

            for i in range(x.shape[0]):
                row = x[i]
                mask = ~np.isnan(row)
                vals = row[mask]

                if vals.size == 0:
                    continue

                order = np.argsort(vals, kind="mergesort")
                sorted_vals = vals[order]

                ranks = np.empty_like(sorted_vals, dtype=float)

                # assign average ranks for ties
                j = 0
                while j < len(sorted_vals):
                    k = j
                    while k < len(sorted_vals) and sorted_vals[k] == sorted_vals[j]:
                        k += 1
                    # pandas ranks start from 1
                    avg_rank = 0.5 * (j + 1 + k)
                    ranks[j:k] = avg_rank
                    j = k

                # invert permutation
                inv = np.empty_like(order)
                inv[order] = np.arange(len(order))

                out[i, mask] = ranks[inv]

            return out

    class cs_zscore(Node):
        def __init__(self, signal): self.signal = signal

        def _compute(self, signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=float)
            out = np.full_like(x, np.nan)

            for i in range(x.shape[0]):
                row = x[i]
                mask = ~np.isnan(row)
                vals = row[mask]

                if vals.size == 0:
                    continue

                mu = np.mean(vals)
                sigma = np.std(vals)
                if sigma <= 0.:
                    sigma = np.nan

                out[i, mask] = (vals - mu) / sigma

            return out

    class add(Node):
        def __init__(self, x, y): self.x, self.y = x, y

        def _compute(self, x, y): return x + y

    class subtract(Node):
        def __init__(self, x, y): self.x, self.y = x, y

        def _compute(self, x, y): return x - y

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
            result = np.full((n, p, p), np.nan, dtype=float)
            for t in range(lookback, n):
                result[t] = np.cov(returns[t-lookback: t+1], rowvar=False)
            return result
