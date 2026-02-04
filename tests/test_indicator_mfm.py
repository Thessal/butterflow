import pytest
import numpy as np
from butterflow import lex, Parser, TypeChecker, Builder, Runtime

def test_full_pipeline_mfm_range():
    """
    Reproduces the original issue:
    1. Parses code with nested functions (range inside mfm).
    2. Runs TypeChecker (verifies scope fix).
    3. Builds Graph (verifies macro expansion).
    4. Runs Runtime (verifies operators and valid output).
    """
    code = """
    range(upper : Signal<Float>, lower : Signal<Float>) : Signal<Float> = {
        # Calcuates range from two boundary. upper - lower
        epsilon : Signal<Float> = const(value=0.000001)
        result : Signal<Float> = add(x=subtract(x=upper, y=lower), y=epsilon)
    }

    # Money‑Flow Multiplier
    mfm(high : Signal<Float>, low : Signal<Float>, close : Signal<Float>) : Signal<Float> = {
        # ((close‑low) – (high‑close)) / (high‑low)
        numerator : Signal<Float> = subtract(x=range(upper=close, lower=low), y=range(upper=high, lower=close))
        price_range : Signal<Float> = range(upper=high, lower=low)
        result : Signal<Float> = divide(dividend=numerator, divisor=price_range)
    }

    result = mfm(high=data(id="high"), low=data(id="low"), close=data(id="close"))
    """

    # 1. Lex
    tokens = lex(code)
    assert len(tokens) > 0

    # 2. Parse
    parser = Parser(tokens)
    stmts = parser.parse()
    assert len(stmts) > 0

    # 3. Type Check
    checker = TypeChecker()
    checker.check(stmts) # Should not raise TypeError

    # 4. Build Graph
    builder = Builder(silent=True)
    graph = builder.build(stmts)
    assert graph is not None

    # 5. Runtime
    length = 10
    # Use deterministic data for testing
    np.random.seed(42)
    high = np.random.uniform(100, 200, length)
    low = high - np.random.uniform(0, 10, length)
    close = low + np.random.uniform(0, high - low, length)
    
    # Mock data loading
    data_map = {
        'data("high")': high,
        'data("low")': low,
        'data("close")': close,
    }
    
    # Depending on how the graph is built, it might request 'data("open")' if implicit dependencies exist, 
    # but strictly from the code above, only high/low/close are needed.
    # However, if const() accesses data("close") internally (as seen in operators.py), ensure it's present.
    
    runtime = Runtime(data=data_map)
    result = runtime.run(graph)
    
    assert result.shape == (length,) or result.shape == (length, 1) or result.ndim == 1 # Adjust based on actual shape output
    
    #Manual Calculation verification
    # range(a, b) = (a - b) + 1e-6
    # num = range(close, low) - range(high, close)
    #     = ((close - low) + 1e-6) - ((high - close) + 1e-6)
    #     = (close - low) - (high - close)  <-- epsilons cancel out in numerator? 
    #     Wait, subtract(x, y). 
    #     x = (close - low) + epsilon
    #     y = (high - close) + epsilon
    #     x - y = close - low - high + close = 2*close - low - high
    #
    # den = range(high, low) = (high - low) + 1e-6
    #
    # result = num / den
    
    epsilon = 1e-6
    range_close_low = (close - low) + epsilon
    range_high_close = (high - close) + epsilon
    numerator = range_close_low - range_high_close
    price_range = (high - low) + epsilon
    
    expected = numerator / price_range
    
    # Check if result matches expected
    # Note: Runtime might return 2D array if inputs are treated as such. 
    # operators.py indicates some 2D logic (N, P). But here inputs are 1D? 
    # Let's check typical behavior. The random data generation in reproduce_issue used 1D but invalid shape caused errors before?
    # In reproduce_issue.py, inputs were (length,). Numpy broadcasting handles it.
    
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    pytest.main([__file__])
