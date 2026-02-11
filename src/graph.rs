use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub result_id: usize, // Index of the final result node in the nodes vector
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Node {
    pub id: usize,
    pub op: String,
    pub args: HashMap<String, ArgValue>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ArgValue {
    NodeId(usize),
    LiteralFloat(f64),
    LiteralInt(i64),
    LiteralString(String),
    LiteralBool(bool),
}
