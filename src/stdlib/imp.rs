use std::collections::HashMap;
use anyhow::{Result, anyhow};
use crate::runtime::RuntimeValue;

pub fn op_const(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    if let Some(val) = args.get("value") {
        Ok(val.clone())
    } else {
        Err(anyhow!("Missing 'value' argument for const"))
    }
}

pub fn op_add(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Float(x + y))
}

// Support both 'sub' and 'subtract'
pub fn op_subtract(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Float(x - y))
}

pub fn op_multiply(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Float(x * y))
}

pub fn op_divide(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let dividend = get_float(&args, "dividend")?;
    let divisor = get_float(&args, "divisor")?;
    if divisor == 0.0 {
        return Err(anyhow!("Division by zero"));
    }
    Ok(RuntimeValue::Float(dividend / divisor))
}

pub fn op_data(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let id_val = args.get("id").ok_or_else(|| anyhow!("Missing 'id' argument for data"))?;
    match id_val {
        RuntimeValue::String(id) => {
             // Mock data source
             match id.as_str() {
                 "high" => Ok(RuntimeValue::Float(105.0)),
                 "low" => Ok(RuntimeValue::Float(95.0)),
                 "close" => Ok(RuntimeValue::Float(100.0)),
                 "volume" => Ok(RuntimeValue::Float(1000.0)),
                 _ => Ok(RuntimeValue::Float(0.0)),
             }
        }
        _ => Err(anyhow!("'id' must be a string")),
    }
}

// Mocks for validation
pub fn op_abs(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    Ok(RuntimeValue::Float(x.abs()))
}

pub fn op_max(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Float(x.max(y)))
}

pub fn op_min(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Float(x.min(y)))
}

pub fn op_greater(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Bool(x > y))
}

pub fn op_less(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Bool(x < y))
}

pub fn op_equals(args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    let x = get_float(&args, "x")?;
    let y = get_float(&args, "y")?;
    Ok(RuntimeValue::Bool((x - y).abs() < f64::EPSILON))
}

// Generic mock for other operators to allow runtime to pass
pub fn op_mock(_args: HashMap<String, RuntimeValue>) -> Result<RuntimeValue> {
    Ok(RuntimeValue::Float(0.0))
}

fn get_float(args: &HashMap<String, RuntimeValue>, name: &str) -> Result<f64> {
     match args.get(name) {
         Some(RuntimeValue::Float(f)) => Ok(*f),
         Some(RuntimeValue::Int(i)) => Ok(*i as f64),
         Some(_) => Err(anyhow!("Argument '{}' is not a number", name)),
         None => Err(anyhow!("Missing argument '{}'", name)),
     }
}
