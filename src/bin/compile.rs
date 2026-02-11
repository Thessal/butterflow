use anyhow::{Result, Context, anyhow};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use butterflow::compiler::parser::parse_stmts;
use butterflow::compiler::typechecker::TypeChecker;
use butterflow::compiler::graph_builder::GraphBuilder;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input source file
    #[arg(required = true)]
    input: PathBuf,

    /// Output graph file (JSON)
    #[arg(short, long, default_value = "output.json")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // 1. Read input
    let code = fs::read_to_string(&args.input).context("Failed to read input file")?;
    
    // 2. Parse
    println!("Parsing...");
    let stmts = parse_stmts(&code).context("Failed to parse")?;
    
    // 3. Type Check
    println!("Type Checking...");
    let mut checker = TypeChecker::new();
    checker.check(&stmts).context("Type checking failed")?;
    
    // 4. Build Graph
    println!("Building Graph...");
    let mut builder = GraphBuilder::new();
    let graph = builder.build(&stmts).context("Failed to build graph")?;
    
    // 5. Serialize
    println!("Serializing...");
    let json = serde_json::to_string_pretty(&graph).context("Failed to serialize graph")?;
    fs::write(&args.output, json).context("Failed to write output file")?;
    
    println!("Done. Graph written to {:?}", args.output);
    Ok(())
}



/*
// Old logic from validate.rs: loading all functions and calculating targets
// This is preserved here for reference.

use std::collections::HashMap;
use std::path::PathBuf;
use butterflow::compiler::{
    parser::parse_stmts,
    typechecker::TypeChecker,
    graph_builder::GraphBuilder,
    ast::{Stmt, FuncDef, Expr, Literal},
};
use std::process::Command;

fn old_validate_logic() -> Result<()> {
    let indicators_dir = "technical_indicators";
    let entries = fs::read_dir(indicators_dir).context("Failed to read directory")?;
    
    // 1. Load all functions
    let mut all_functions: HashMap<String, FuncDef> = HashMap::new();
    let mut file_main_funcs: Vec<(PathBuf, String)> = Vec::new(); // (path, func_name)

    println!("Loading functions...");
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("butter") {
            // Skip test files
            if path.file_stem().and_then(|s| s.to_str()).map(|s| s.ends_with("_test")).unwrap_or(false) {
                continue;
            }

            let content = fs::read_to_string(&path)?;
            match parse_stmts(&content) {
                Ok(stmts) => {
                    for stmt in stmts {
                        if let Stmt::FuncDef(f) = stmt {
                            all_functions.insert(f.name.clone(), f.clone());
                            
                            // Heuristic: if function name matches filename, it's the main function
                            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                                if f.name == stem {
                                    file_main_funcs.push((path.clone(), f.name.clone()));
                                }
                            }
                        }
                    }
                },
                Err(e) => eprintln!("Failed to parse {}: {}", path.display(), e),
            }
        }
    }
    
    println!("Loaded {} unique functions.", all_functions.len());
    println!("Identified {} main validation targets.", file_main_funcs.len());

    let mut success_count = 0;
    
    // 2. Validate each target
    for (path, func_name) in &file_main_funcs {
        println!("Validating {} ({})", path.display(), func_name);
        
        // Build Program: All functions + Result Assignment
        let mut program_stmts: Vec<Stmt> = all_functions.values().cloned().map(Stmt::FuncDef).collect();
        
        // Construct args
        let main_func = all_functions.get(func_name).unwrap();
        let mut call_args = HashMap::new();
        
        for (arg_name, arg_type_str) in &main_func.args {
             let expr = if arg_type_str.contains("Signal") || arg_type_str == "Float" {
                 // data(id="name")
                 // let mut args_map = HashMap::new(); // HashMap<String, Expr>
                 // args_map.insert("id".to_string(), Expr::Literal(Literal::String(arg_name.clone())));
                 // Expr::Call("data".to_string(), args_map)
                 // NOTE: Creating HashMap manually as in original code
                 Expr::Literal(Literal::Float(0.0)) // simplified for comment
             } else {
                 Expr::Literal(Literal::Float(0.5))
             };
             call_args.insert(arg_name.clone(), expr);
        }
        
        // ... (rest of logic omitted/simplified for brevity in comment)
    }
    Ok(())
}
*/
