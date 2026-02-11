use crate::typesystem::Type;
use std::collections::HashMap;

pub fn std_lib() -> HashMap<String, Type> {
    let mut m = HashMap::new();
    
    // Helper to create atomic Float
    let float = || Type::Atomic("Float".to_string());
    let int = || Type::Atomic("Int".to_string());
    let signal_float = || Type::Generic("Signal".to_string(), Box::new(float()));
    // For simplicity in this validation phase, we treat inputs as either Signal, atomic Float, or Int (converted)
    let signal_float_or_float = || Type::Either(vec![signal_float(), float(), int()]);

    // data(id: String) -> Signal<Float>
    m.insert("data".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("id".to_string(), Type::Atomic("String".to_string()))
        ]))),
        Box::new(signal_float())
    ));
    
    // const(value: Float) -> Signal<Float>
    m.insert("const".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("value".to_string(), float())
        ]))),
        Box::new(signal_float())
    ));

    // add(x: Signal|Float, y: Signal|Float) -> Signal<Float>
    m.insert("add".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
            ("y".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // subtract(x: Signal|Float, y: Signal|Float) -> Signal<Float>
    m.insert("subtract".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
            ("y".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // multiply(x: Signal|Float, y: Signal|Float) -> Signal<Float>
    m.insert("multiply".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
            ("y".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // divide(dividend: Signal|Float, divisor: Signal|Float) -> Signal<Float>
    m.insert("divide".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("dividend".to_string(), signal_float_or_float()),
            ("divisor".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // abs(x: Signal|Float) -> Signal<Float>
    // Supporting both 'x' and 'signal' argument names
    let abs_x = Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    );
    let abs_signal = Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    );
    m.insert("abs".to_string(), Type::Either(vec![abs_x, abs_signal]));
    
    // max(x: Signal|Float, y: Signal|Float) -> Signal<Float>
    m.insert("max".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
            ("y".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
     // min(x: Signal|Float, y: Signal|Float) -> Signal<Float>
    m.insert("min".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
            ("y".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // greater(signal: Signal|Float, thres: Signal|Float) -> Signal<Float>
    m.insert("greater".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float_or_float()),
            ("thres".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // less(signal: Signal|Float, thres: Signal|Float) -> Signal<Float>
    m.insert("less".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float_or_float()),
            ("thres".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // equals(x: Signal|Float, y: Signal|Float) -> Signal<Float>
    m.insert("equals".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
            ("y".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // Comparison Aliases (if used in examples)
    m.insert("gt".to_string(), m.get("greater").unwrap().clone());
    m.insert("lt".to_string(), m.get("less").unwrap().clone());
    
    // tradewhen(signal: Signal, enter: Signal|Int, exit: Signal|Int, period: Int) -> Signal
    m.insert("tradewhen".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("enter".to_string(), signal_float_or_float()), 
            ("exit".to_string(), signal_float_or_float()),
            ("period".to_string(), int()),
        ]))),
        Box::new(signal_float())
    ));
    
    // shift(signal: Signal, period: Int) -> Signal
    m.insert("shift".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), int()),
        ]))),
        Box::new(signal_float())
    ));
    
    // clip(signal: Signal, min: Float, max: Float) -> Signal
    m.insert("clip".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("min".to_string(), float()),
            ("max".to_string(), float()),
        ]))),
        Box::new(signal_float())
    ));

    // Time Series Operators
    
    // ts_mean(signal: Signal, period: Int) -> Signal
    m.insert("ts_mean".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()), // Accepting float/signal for flexible period? Or just int? Examples often pass Int.
        ]))),
        Box::new(signal_float())
    ));
    
    // ts_sum(signal: Signal, period: Int) -> Signal
    m.insert("ts_sum".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // ts_max(signal: Signal, period: Int) -> Signal
    m.insert("ts_max".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // ts_min(signal: Signal, period: Int) -> Signal
    m.insert("ts_min".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // ts_decay_linear(signal: Signal, period: Int) -> Signal
    m.insert("ts_decay_linear".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // ts_diff(signal: Signal, period: Int)
     m.insert("ts_diff".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // ts_delay(signal: Signal, period: Int)
     m.insert("ts_delay".to_string(), Type::Operator(
         Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // ratio(x, y)
    m.insert("ratio".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("x".to_string(), signal_float_or_float()),
            ("y".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // ts_argminmax(signal: Signal, period: Int) -> Signal
    m.insert("ts_argminmax".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
             ("signal".to_string(), signal_float()),
             ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // ema(signal, period) -> ts_decay_exp signature
    m.insert("ema".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // ts_decay_exp
    m.insert("ts_decay_exp".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
            ("signal".to_string(), signal_float()),
            ("period".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));

    // where(condition: Signal, val_true: Signal, val_false: Signal) -> Signal
    m.insert("where".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
             ("condition".to_string(), signal_float()),
             ("val_true".to_string(), signal_float_or_float()),
             ("val_false".to_string(), signal_float_or_float()),
        ]))),
        Box::new(signal_float())
    ));
    
    // ts_ffill
    m.insert("ts_ffill".to_string(), Type::Operator(
        Box::new(Type::Dict(HashMap::from([
             ("signal".to_string(), signal_float()),
             ("period".to_string(), int()),
        ]))),
        Box::new(signal_float())
    ));

    m
}
