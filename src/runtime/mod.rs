pub mod engine;


#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
}
