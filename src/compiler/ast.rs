use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Stmt {
    FuncDef(FuncDef),
    Assign(Assign),
}

#[derive(Debug, Clone)]
pub struct FuncDef {
    pub name: String,
    pub args: HashMap<String, String>, // Name -> Type
    pub return_type: String,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub assigns: Vec<Assign>,
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub target: (String, Option<String>), // Name, Type
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    VarRef(String),
    Call(String, HashMap<String, Expr>),
    Block(Block),
}

#[derive(Debug, Clone)]
pub enum Literal {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
}
