use crate::compiler::ast::{Stmt, FuncDef, Assign, Expr, Literal, Block};
use crate::graph::{Graph, Node, ArgValue};
use std::collections::HashMap;
use anyhow::{Result, anyhow};

pub struct GraphBuilder {
    nodes: Vec<Node>,
    scope: HashMap<String, usize>, // Variable name -> Node ID
    macros: HashMap<String, FuncDef>,
    silent: bool,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            scope: HashMap::new(),
            macros: HashMap::new(),
            silent: false,
        }
    }

    pub fn build(&mut self, stmts: &[Stmt]) -> Result<Graph> {
        for stmt in stmts {
            match stmt {
                Stmt::FuncDef(f_def) => {
                    self.macros.insert(f_def.name.clone(), f_def.clone());
                }
                Stmt::Assign(assign) => {
                    let node_id_or_val = self.eval_expr(&assign.expr, &self.scope.clone())?;
                    let (name, _) = &assign.target;
                    
                    match node_id_or_val {
                        EvalResult::NodeId(id) => {
                            self.scope.insert(name.clone(), id);
                        }
                        EvalResult::Value(val) => {
                            // If it's a value, wrap it in a const node? 
                            // Or just keep track of it as a literal?
                            // For simplicity, let's treat top-level variables as pointing to Nodes.
                            // If it's a literal, create a "const" node if needed, 
                            // OR just store the value in scope?
                            // Issue: if used as an argument, it might need to be a node.
                            // BUT arguments in Graph can be Literals.
                            
                            // Let's create a specific node wrapper if it is assigned to a variable,
                            // OR we can just store the literal in a "const" node.
                            // Operators like 'add' expect Signal (Node) or Float.
                            // If we pass a literal to 'add', we handle it in eval_call.
                            
                            // If we assign `x = 1.0`, then `y = add(x, 2.0)`.
                            // `x` should probably be a const node so it has an ID.
                            let id = self.create_const_node(val);
                            self.scope.insert(name.clone(), id);
                        }
                    }
                }
            }
        }

        let result_id = *self.scope.get("result").ok_or_else(|| anyhow!("No 'result' variable found in global scope"))?;

        Ok(Graph {
            nodes: self.nodes.clone(),
            result_id,
        })
    }
    
    fn create_const_node(&mut self, val: ArgValue) -> usize {
        // Create a 'const' operator node
        let id = self.nodes.len();
        // 'const' op expects 'value' arg
        let mut args = HashMap::new();
        args.insert("value".to_string(), val);
        
        self.nodes.push(Node {
            id,
            op: "const".to_string(),
            args,
        });
        id
    }

    fn eval_expr(&mut self, expr: &Expr, scope: &HashMap<String, usize>) -> Result<EvalResult> {
        match expr {
            Expr::Literal(lit) => Ok(EvalResult::Value(match lit {
                Literal::Float(f) => ArgValue::LiteralFloat(*f),
                Literal::Int(i) => ArgValue::LiteralInt(*i),
                Literal::String(s) => ArgValue::LiteralString(s.clone()),
                Literal::Bool(b) => ArgValue::LiteralBool(*b),
            })),
            Expr::VarRef(name) => {
                 if let Some(&id) = scope.get(name) {
                     Ok(EvalResult::NodeId(id))
                 } else {
                     Err(anyhow!("Undefined variable '{}'", name))
                 }
            },
            Expr::Call(name, args) => {
                // Check if macro
                if let Some(func_def) = self.macros.get(name).cloned() {
                    // Macro expansion
                    // 1. Evaluate arguments
                    let mut eval_args = HashMap::new();
                    for (k, v) in args {
                        eval_args.insert(k.clone(), self.eval_expr(v, scope)?);
                    }
                    
                    // 2. Prepare local scope
                    let mut local_scope = HashMap::new();
                    // Map named args
                    for (arg_name, _type) in &func_def.args {
                        if let Some(val) = eval_args.get(arg_name) {
                             match val {
                                 EvalResult::NodeId(id) => { local_scope.insert(arg_name.clone(), *id); },
                                 EvalResult::Value(v) => {
                                     // Create const node for literal arg passed to function
                                     let id = self.create_const_node(v.clone());
                                     local_scope.insert(arg_name.clone(), id);
                                 }
                             }
                        } else {
                             return Err(anyhow!("Missing argument '{}' for macro '{}'", arg_name, name));
                        }
                    }
                    
                    // 3. Evaluate body
                    self.eval_block(&func_def.body, &mut local_scope)
                } else {
                    // Standard Operator Call
                    let mut op_args = HashMap::new();
                    
                    for (k, v) in args {
                        let res = self.eval_expr(v, scope)?;
                        match res {
                            EvalResult::NodeId(nid) => {
                                op_args.insert(k.clone(), ArgValue::NodeId(nid));
                            }
                            EvalResult::Value(val) => {
                                op_args.insert(k.clone(), val);
                            }
                        }
                    }
                    
                    let id = self.nodes.len();
                    self.nodes.push(Node {
                        id,
                        op: name.clone(),
                        args: op_args,
                    });
                    
                    Ok(EvalResult::NodeId(id))
                }
            },
            Expr::Block(block) => {
                 let mut inner_scope = scope.clone();
                 self.eval_block(block, &mut inner_scope)
            }
        }
    }
    
    fn eval_block(&mut self, block: &Block, scope: &mut HashMap<String, usize>) -> Result<EvalResult> {
        let mut last_res: Option<EvalResult> = None;
        
        for assign in &block.assigns {
            let res = self.eval_expr(&assign.expr, scope)?;
            let (name, _) = &assign.target;
            
            let id = match res {
                EvalResult::NodeId(id) => {
                    scope.insert(name.clone(), id);
                    id
                },
                EvalResult::Value(v) => {
                     let id = self.create_const_node(v);
                     scope.insert(name.clone(), id);
                     id
                }
            };
            last_res = Some(EvalResult::NodeId(id));
        }
        
        if let Some(&id) = scope.get("result") {
            Ok(EvalResult::NodeId(id))
        } else if let Some(res) = last_res {
            Ok(res)
        } else {
             Ok(EvalResult::Value(ArgValue::LiteralBool(false)))
        }
    }
}

enum EvalResult {
    NodeId(usize),
    Value(ArgValue),
}
