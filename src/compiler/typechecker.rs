use crate::typesystem::Type;
use crate::compiler::ast::{Stmt, FuncDef, Assign, Expr, Literal, Block};
use crate::stdlib::registry;
use std::collections::HashMap;
use anyhow::{Result, anyhow};

pub struct TypeChecker {
    symbol_table: HashMap<String, Type>,
    func_signatures: HashMap<String, Type>,
    silent: bool,
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            symbol_table: HashMap::from([
                ("nan".to_string(), Type::Atomic("Float".to_string())),
            ]),
            func_signatures: registry::std_lib(),
            silent: false,
        }
    }

    pub fn check(&mut self, stmts: &[Stmt]) -> Result<()> {
        // Pass 1: Register all function signatures
        for stmt in stmts {
            if let Stmt::FuncDef(f) = stmt {
                let mut arg_types = HashMap::new();
                for (name, type_str) in &f.args {
                    arg_types.insert(name.clone(), parse_type_str(type_str));
                }
                let ret_type = parse_type_str(&f.return_type);
                let func_type = Type::Operator(
                    Box::new(Type::Dict(arg_types)),
                    Box::new(ret_type)
                );
                self.func_signatures.insert(f.name.clone(), func_type);
            }
        }

        // Pass 2: Check bodies and assignments
        for stmt in stmts {
            match stmt {
                Stmt::FuncDef(f_def) => self.check_func_def(f_def)?,
                Stmt::Assign(assign) => self.check_assign(assign)?,
            }
        }
        Ok(())
    }

    fn check_func_def(&mut self, f: &FuncDef) -> Result<()> {
        // Construct function type
        // For simplicity, assuming named arguments in Dict type for now,
        // but AST has args as HashMap<String, String>.
        
        // 1. Parse argument types
        let mut arg_types = HashMap::new();
        for (name, type_str) in &f.args {
            arg_types.insert(name.clone(), parse_type_str(type_str));
        }
        
        let ret_type = parse_type_str(&f.return_type);
        
        let func_type = Type::Operator(
            Box::new(Type::Dict(arg_types.clone())),
            Box::new(ret_type.clone())
        );
        
        self.func_signatures.insert(f.name.clone(), func_type);
        
        // 2. Check body
        // Create new scope with args
        let mut inner_checker = TypeChecker::new();
        inner_checker.func_signatures = self.func_signatures.clone(); // inherit functions
        inner_checker.symbol_table = arg_types; // args are in scope
        
        inner_checker.check_block(&f.body)?;
        
        // If last statement in block is assignment to "result", check its type against ret_type
        // Or if simple expression block (which we converted to block with "result" assignment)
        
        // We need to verify that 'result' variable in inner scope matches return type
        if let Some(res_type) = inner_checker.symbol_table.get("result") {
            if !ret_type.matches(res_type) {
                 return Err(anyhow!("Function '{}' return type mismatch. Expected {}, got {}", f.name, ret_type, res_type));
            }
        } else {
             // If void function? Butterflow usually returns Signal.
             // If body doesn't assign result, it's an error?
             return Err(anyhow!("Function '{}' must assign to 'result'", f.name));
        }
        
        Ok(())
    }

    fn check_assign(&mut self, assign: &Assign) -> Result<()> {
        let rhs_type = self.infer_type(&assign.expr)?;
        let (name, decl_type_str) = &assign.target;
        
        if let Some(s) = decl_type_str {
            let decl_type = parse_type_str(s);
            if !decl_type.matches(&rhs_type) {
                return Err(anyhow!("Type Mismatch for '{}': Expected {}, got {}", name, decl_type, rhs_type));
            }
        }
        
        self.symbol_table.insert(name.clone(), rhs_type);
        Ok(())
    }
    
    fn check_block(&mut self, block: &Block) -> Result<()> {
        for assign in &block.assigns {
            self.check_assign(assign)?;
        }
        Ok(())
    }

    fn infer_type(&self, expr: &Expr) -> Result<Type> {
        match expr {
            Expr::Literal(lit) => Ok(match lit {
                Literal::Float(_) => Type::Atomic("Float".to_string()),
                Literal::Int(_) => Type::Atomic("Int".to_string()),
                Literal::Bool(_) => Type::Atomic("Bool".to_string()),
                Literal::String(_) => Type::Atomic("String".to_string()),
            }),
            Expr::VarRef(name) => {
                self.symbol_table.get(name)
                    .cloned()
                    .ok_or_else(|| anyhow!("Undefined variable '{}'", name))
            },
            Expr::Call(name, args) => {
                let func_type = self.func_signatures.get(name)
                    .ok_or_else(|| anyhow!("Unknown function '{}'", name))?;
                
                // Helper to check a single signature (Operator)
                let check_sig = |req_args_box: &Box<Type>, ret_type_box: &Box<Type>| -> Result<Type> {
                    if let Type::Dict(req_fields) = &**req_args_box {
                       let mut supplied_types = HashMap::new();
                       for (k, v) in args {
                           supplied_types.insert(k.clone(), self.infer_type(v)?);
                       }
                       
                       for (req_k, req_t) in req_fields {
                           if let Some(sup_t) = supplied_types.get(req_k) {
                               if !req_t.matches(sup_t) {
                                   if !self.silent {
                                       println!("Type mismatch for arg '{}': expected {}, got {}", req_k, req_t, sup_t);
                                       if let Type::Either(opts) = req_t {
                                            for opt in opts {
                                                println!("  - Option: {} matches {}? {}", opt, sup_t, opt.matches(sup_t));
                                            }
                                       }
                                   }
                                   return Err(anyhow!("Argument '{}' type mismatch", req_k));
                               }
                           } else {
                               return Err(anyhow!("Missing argument '{}'", req_k));
                           }
                       }
                       Ok(*ret_type_box.clone())
                    } else {
                        Err(anyhow!("Non-dict args not supported"))
                    }
                };

                match func_type {
                    Type::Operator(req, ret) => {
                        check_sig(req, ret).map_err(|e| anyhow!("In call to '{}': {}", name, e))
                    },
                    Type::Either(variants) => {
                        // Try each variant. If one succeeds, return it.
                        // If all fail, return error describing why.
                        let mut last_err = anyhow!("No matching signature for '{}'", name);
                        for variant in variants {
                            if let Type::Operator(req, ret) = variant {
                                match check_sig(req, ret) {
                                    Ok(t) => return Ok(t),
                                    Err(e) => last_err = e,
                                }
                            }
                        }
                        Err(anyhow!("No matching signature for '{}'. Last error: {}", name, last_err))
                    },
                    _ => Err(anyhow!("'{}' is not a function/operator", name))
                }
            },
            Expr::Block(b) => {
                 // Block evaluates to the Dictionary of its assignments? 
                 // Or "result" field?
                 // In Python parser: parse_block returns Block(Expr)
                 // In Python typechecker: Block -> DictType of all assigned fields.
                 
                 // If we strictly follow python logic:
                 let mut fields: HashMap<String, Type> = HashMap::new();
                 // We need to infer types of assignments without polluting current scope PERMANENTLY? 
                 // Actually blocks (like in `range = { ... }`) create a struct/dict.
                 // So we need a temporary scope.
                 // But wait, `check_block` above was for function body (in-place).
                 // `infer_type` for Expr::Block means it's used as a value (Struct/Scope).
                 
                 // Let's create a temporary checker or scope
                 // Since we don't have nested scope support in this struct easily (no stack), 
                 // let's just clone.
                 let mut inner_checker = TypeChecker::new(); 
                 inner_checker.symbol_table = self.symbol_table.clone(); // Capture outer
                 inner_checker.func_signatures = self.func_signatures.clone();
                 
                 inner_checker.check_block(b)?;
                 
                 // Now extract *new* variables as fields
                 // Actually python logic: fields = {k: infer(v) for k,v in expr.assigns}
                 // So it returns a DictType of the items defined *inside* the block.
                 
                 let mut dict_fields = HashMap::new();
                 for assign in &b.assigns {
                     let (name, _) = &assign.target;
                     let t = inner_checker.symbol_table.get(name).unwrap().clone();
                     dict_fields.insert(name.clone(), t);
                 }
                 Ok(Type::Dict(dict_fields))
            }
        }
    }
}

// Helpers
fn parse_type_str(s: &str) -> Type {
    let s = s.trim();
    // Very basic parsing for "Signal<Float>", "Float", etc.
    // In reality, this should parse the structure properly.
    // For now, handling Atomic and simple Generic "Signal<...>"
    
    if s.starts_with("Signal<") && s.ends_with(">") {
        let inner = &s[7..s.len()-1];
        Type::Generic("Signal".to_string(), Box::new(parse_type_str(inner)))
    } else {
        Type::Atomic(s.to_string())
    }
}
