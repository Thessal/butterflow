import sys
sys.path.append("./src/butterflow")

import pytest
import numpy as np 
from operators import STD_LIB, STD_LIB_IMPL, Operator, Either, Generic, Atomic
import itertools

N = 20
N1 = 10
P = 20
np.random.seed(seed=42)

def build_random_data_pair():
    result = dict()
    matf1 = np.random.rand(N,P)
    matf2 = matf1.copy()
    matf2[N1:] = np.random.rand(N-N1,P)
    f = float(np.random.normal(0,1))
    i = int(np.random.randint(-10,10))
    result[repr(Generic("Signal", Atomic("Float")))] = (matf1, matf2)
    result[repr(Atomic("Float"))] = (f,f)
    result[repr(Atomic("Int"))] = (i,i)
    return result

def parse_patterns(args):
    names = list(args.keys())
    types = [(v.options if isinstance(v, Either) else [v]) for v in args.values()]
    matches = [dict(zip(names,t)) for t in itertools.product(*types)]
    return matches

def run_pattern(op, pattern, data):
    paramsA = dict()
    paramsB = dict()
    for name, dtype in pattern.items():
        paramsA[name], paramsB[name] = data[repr(dtype)]
    resultA = op(**paramsA).compute(cache=None)
    resultB = op(**paramsB).compute(cache=None)
    return resultA, resultB

def forward_looking_test(op_name):
    op_info = STD_LIB[op_name]
    op = STD_LIB_IMPL.get(op_name)
    # op_info = op
    args = op_info.args.fields
    patterns = parse_patterns(args)
    data = build_random_data_pair()
    errors = []
    for pattern in patterns:
        result_A, result_B = run_pattern(op, pattern, data)
        if type(result_A) == type(result_B) == np.ndarray:
            result_A = result_A[:N1]
            result_B = result_B[:N1]
        if not np.array_equal(result_A, result_B, equal_nan=True):
            # print("A\n", result_A[-3:, :5], "\nB\n", result_B[-3: :5])
            # ValueError(f"Forward looking bias in {op_name}, {pattern}")
            errors.append(f"Forward looking bias in {op_name}, {pattern}")   
    return errors

def test_forward_looking():
    op_names = STD_LIB.keys()
    errors = []
    for op_name in op_names:
        if op_name != "data":
            err = forward_looking_test(op_name)
            errors.extend(err)
    if errors:
        print("\n".join(errors))
        assert not errors

if __name__=="__main__":
    pytest.main()