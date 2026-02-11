use pest::Parser;
use pest_derive::Parser;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "compiler/grammar.pest"]
pub struct ButterflowParser;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    Ident(String),
    Op(String),
}

#[derive(Error, Debug)]
pub enum LexerError {
    #[error("Parsing error: {0}")]
    ParseError(#[from] pest::error::Error<Rule>),
}

pub fn lex(input: &str) -> Result<Vec<Token>, LexerError> {
    // Pest handles tokenization during parsing, but if we need a separate lexer step
    // we can iterate over pairs. For now, since we are using Pest, 
    // we might just skip to parsing or just verify we can tokenize.
    // Let's implement a simple tokenizer wrapper if needed, 
    // or rely on Pest pairs for the Parser phase.
    
    // For this port, we will integrate Lexing into Parsing via Pest.
    // This function is just a placeholder or could return a stream of tokens if we weren't using Pest.
    Ok(vec![]) 
}
