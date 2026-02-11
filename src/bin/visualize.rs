use anyhow::{Result, Context};
use clap::Parser;
use std::fs;
use butterflow::graph::{Graph, ArgValue};

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
    
    println!("v{} = Graph Result", graph.result_id);
    println!("----------------------------------------");

    for node in &graph.nodes {
        let mut args_str_parts = Vec::new();
        // Sort args by key for stable output
        let mut sorted_args: Vec<_> = node.args.iter().collect();
        sorted_args.sort_by_key(|(k, _)| *k);

        for (k, v) in sorted_args {
            let val_str = match v {
                ArgValue::NodeId(id) => format!("v{}", id),
                ArgValue::LiteralFloat(f) => format!("{:.6}", f),
                ArgValue::LiteralInt(i) => format!("{}", i),
                ArgValue::LiteralString(s) => format!("\"{}\"", s),
                ArgValue::LiteralBool(b) => format!("{}", b),
            };
            args_str_parts.push(format!("{}={}", k, val_str));
        }
        
        println!("v{} = {}({})", node.id, node.op, args_str_parts.join(", "));
    }

    Ok(())
}
