use anyhow::{Result, Context};
use clap::Parser;
use std::fs;
use butterflow::graph::Graph;
use butterflow::runtime::engine;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input graph JSON file
    #[arg(required = true)]
    input: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let content = fs::read_to_string(&args.input).context("Failed to read input file")?;
    let graph: Graph = serde_json::from_str(&content).context("Failed to parse graph JSON")?;
    
    match engine::run(graph) {
        Ok(val) => println!("Final Result: {:?}", val),
        Err(e) => eprintln!("Execution Error: {}", e),
    }

    Ok(())
}
