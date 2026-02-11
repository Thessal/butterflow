use anyhow::{Result, Context, anyhow};
use std::fs;
use std::path::PathBuf; // PathBuf needed
use std::collections::HashMap; // HashMap needed

use butterflow::compiler::{
    parser::parse_stmts,
    typechecker::TypeChecker,
    graph_builder::GraphBuilder,
    ast::{Stmt, FuncDef, Expr, Literal}, // AST items
};

// Removed regex and std::process::Command (except maybe for visualization if I kept it binary? 
// No, I can use internal types, but visualize is a binary. 
// I can copy visualize logic or just execute the binary. 
// Executing binary is easier for now to keep code coupled loosely.)
use std::process::Command;

fn main() -> Result<()> {
    let indicators_dir = "technical_indicators";
    let entries = fs::read_dir(indicators_dir).context("Failed to read directory")?;
    
    let mut success_count = 0;
    let mut total_files = 0;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("butter") {
            // Skip test files
            if path.file_stem().and_then(|s| s.to_str()).map(|s| s.ends_with("_test")).unwrap_or(false) {
                continue;
            }

            println!("Processing {}", path.display());

            let content = fs::read_to_string(&path)?;
            match parse_stmts(&content) {
                Ok(mut stmts) => {
                    // Find main function for this file
                    let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                    let mut main_func_opt = None;

                    for stmt in &stmts {
                        if let Stmt::FuncDef(f) = stmt {
                            if f.name == file_stem {
                                main_func_opt = Some(f.clone());
                                break;
                            }
                        }
                    }

                    if let Some(main_func) = main_func_opt {
                        total_files += 1;
                        println!("Validating target: {}", main_func.name);
                        
                        // Construct args
                        let mut call_args = HashMap::new();
                        for (arg_name, arg_type_str) in &main_func.args {
                             let expr = if arg_type_str.contains("Signal") || arg_type_str == "Float" {
                                 // data(id="name")
                                 let mut args_map = HashMap::new(); // HashMap<String, Expr>
                                 args_map.insert("id".to_string(), Expr::Literal(Literal::String(arg_name.clone())));
                                 Expr::Call("data".to_string(), args_map)
                             } else if arg_type_str == "Int" {
                                 Expr::Literal(Literal::Int(14))
                             } else if arg_type_str == "String" {
                                 Expr::Literal(Literal::String("mock".to_string()))
                             } else {
                                 Expr::Literal(Literal::Float(0.5))
                             };
                             call_args.insert(arg_name.clone(), expr);
                        }
                        
                        let call_expr = Expr::Call(main_func.name.clone(), call_args);
                        
                        let assign_stmt = Stmt::Assign(butterflow::compiler::ast::Assign {
                            target: ("result".to_string(), None),
                            expr: call_expr,
                        });
                        
                        stmts.push(assign_stmt);
                        
                        // Compile (In-Memory)
                        if let Err(e) = compile_and_visualize(&stmts, &path) {
                            eprintln!("Failed to validate {}: {:?}", path.display(), e);
                        } else {
                            success_count += 1;
                        }
                    } else {
                         println!("Skipping {} (no main function matching filename)", path.display());
                    }
                },
                Err(e) => eprintln!("Failed to parse {}: {}", path.display(), e),
            }
        }
    }
    
    println!("Validation complete: {}/{} successful.", success_count, total_files);
    
    if total_files > 0 && success_count < total_files {
        Err(anyhow!("Some validations failed"))
    } else {
        Ok(())
    }
}

fn compile_and_visualize(stmts: &[Stmt], original_path: &std::path::Path) -> Result<()> {
    // Type Check
    let mut checker = TypeChecker::new();
    checker.check(stmts).context("Type checking failed")?;
    
    // Build Graph
    let mut builder = GraphBuilder::new();
    let graph = builder.build(stmts).context("Graph build failed")?;
    
    // Serialize
    let json = serde_json::to_string_pretty(&graph)?;
    let json_path = original_path.with_extension("json");
    fs::write(&json_path, &json)?;
    
    // Visualize (call binary)
    let eqn_path = original_path.with_extension("eqn");
    let output = Command::new("cargo")
        .args(&["run", "-q", "--bin", "butterflow-visualize", "--", json_path.to_str().unwrap()])
        .output()?;
        
    if !output.status.success() {
        return Err(anyhow!("Visualization failed"));
    }
    fs::write(eqn_path, output.stdout)?;
    
    Ok(())
}
