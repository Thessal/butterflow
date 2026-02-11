use crate::graph::{Graph, Node, ArgValue};
use crate::stdlib::imp;
use crate::runtime::RuntimeValue;
use anyhow::{Result, anyhow};
use std::collections::HashMap;

pub fn run(graph: Graph) -> Result<RuntimeValue> {
    let mut results: HashMap<usize, RuntimeValue> = HashMap::new();

    // Nodes are topologically sorted by construction (append-only)
    // So we can iterate and execute them in order.
    for node in &graph.nodes {
        let val = execute_node(node, &results)?;
        results.insert(node.id, val);
    }

    results.get(&graph.result_id)
        .cloned()
        .ok_or_else(|| anyhow!("Result node {} not found", graph.result_id))
}

fn execute_node(node: &Node, results: &HashMap<usize, RuntimeValue>) -> Result<RuntimeValue> {
    let mut args: HashMap<String, RuntimeValue> = HashMap::new();
    
    for (name, arg_val) in &node.args {
        let val = match arg_val {
            ArgValue::NodeId(id) => results.get(id).ok_or_else(|| anyhow!("Node {} dependency not found", id))?.clone(),
            ArgValue::LiteralFloat(f) => RuntimeValue::Float(*f),
            ArgValue::LiteralInt(i) => RuntimeValue::Int(*i),
            ArgValue::LiteralString(s) => RuntimeValue::String(s.clone()),
            ArgValue::LiteralBool(b) => RuntimeValue::Bool(*b),
        };
        args.insert(name.clone(), val);
    }

    match node.op.as_str() {
        "const" => imp::op_const(args),
        "add" => imp::op_add(args),
        "subtract" | "sub" => imp::op_subtract(args),
        "multiply" => imp::op_multiply(args),
        "divide" => imp::op_divide(args),
        "data" => imp::op_data(args),
        "abs" => imp::op_abs(args),
        "max" => imp::op_max(args),
        "min" => imp::op_min(args),
        "greater" | "gt" => imp::op_greater(args),
        "less" | "lt" => imp::op_less(args),
        "equals" => imp::op_equals(args),
        // Mock any other known operator for validation purposes
        "tradewhen" | "shift" | "clip" | "ts_mean" | "ts_sum" | 
        "ts_max" | "ts_min" | "ts_decay_linear" | "ts_diff" | 
        "ts_delay" | "ts_argminmax" | "ts_decay_exp" | 
        "ts_ffill" | "ts_mae" | "ts_mean_exponential" | 
        "ts_rank" | "ts_std" | "ts_zscore" | "where" | 
        "bool_ffill" | "trigger" | "stoch" | "stoch_k" | "stoch_d" |
        "stoch_rsi" | "stoch_rsi_triggered" | "rsi" | "rsi_divergence" |
        "mfi" | "kdj_wrapper" | "cmf" | "chaikin_oscillator" | 
        "chaikin_volatility" | "awesome_oscillator" | "atr" |
        "aroon" | "adl" | "mfm" | "range" | "tr" | "true_range" |
        "typical_price" | "upper_channel" | "lower_channel" | "mid" |
        "heikin_ashi_close" | "heikin_ashi_high" | "heikin_ashi_low" | "heikin_ashi_open" |
        "is_peak" | "is_trough" | "ratio" | "williams_r" | 
        "lower_shadow" | "upper_shadow" | "body" => imp::op_mock(args),
        _ => Err(anyhow!("Unknown operator: {}", node.op)),
    }
}
