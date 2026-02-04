
import pytest
import numpy as np
from butterflow.operators import STD_LIB_IMPL

def test_tradewhen_logic_and_inplace_safety():
    """
    Verifies that tradewhen:
    1. Correctly follows enter/exit logic
    2. Does NOT modify the input signal in-place
    """
    # 5 time steps, 1 element
    signal = np.array([[10.], [20.], [30.], [40.], [50.]])
    signal_copy = signal.copy()
    
    enter = np.array([[0.], [1.], [0.], [0.], [0.]]) # Enter at t=1 (value 20)
    exit = np.array([[0.], [0.], [0.], [1.], [0.]])  # Exit at t=3
    period = 10
    
    # Expected behavior:
    # t=0: nan (default)
    # t=1: enter=1 -> take signal value 20.
    # t=2: enter=0, exit=0 -> hold value 20.
    # t=3: exit=1 -> becomes nan (or keeps value then becomes nan next step? check logic)
    # Logic: exit cancels hold. If exit is true, output should be nan? 
    # Or strict definition: value is discarded TO nan. So result[3] should be nan.
    
    # Manually constructing the node
    node = STD_LIB_IMPL.tradewhen(signal=None, enter=None, exit=None, period=period)
    node._cache = {} # Mock cache
    
    # Call _compute 
    result = node._compute(signal, enter, exit, period)
    
    # 1. Check In-Place Modification
    np.testing.assert_array_equal(signal, signal_copy, err_msg="Input signal was modified in-place!")
    
    # 2. Check Logic
    # t=0: nan
    assert np.isnan(result[0,0]), "t=0 should be nan"
    
    # t=1: value 20
    assert result[1,0] == 20., f"t=1 should match enter value 20, got {result[1,0]}"
    
    # t=2: hold 20
    assert result[2,0] == 20., f"t=2 should hold value 20, got {result[2,0]}"
    
    # t=3: exit -> nan
    assert np.isnan(result[3,0]), "t=3 should be nan due to exit"
    
    # t=4: still nan
    assert np.isnan(result[4,0]), "t=4 should remain nan"

if __name__ == "__main__":
    pytest.main([__file__])
