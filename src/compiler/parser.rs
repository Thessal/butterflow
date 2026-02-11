
use crate::compiler::ast::{Assign, Block, Expr, FuncDef, Literal, Stmt};
use crate::compiler::lexer::{ButterflowParser, Rule};
use pest::Parser;
use std::collections::HashMap;
use pest::iterators::Pair;
use anyhow::{Result, anyhow};

pub fn parse_stmts(input: &str) -> Result<Vec<Stmt>> {
    let mut pairs = ButterflowParser::parse(Rule::program, input)?;
    let root = pairs.next().ok_or(anyhow!("No root pair"))?;
    
    let mut stmts = Vec::new();
    // 'program' rule has inner 'stmt' rules
    for stmt_pair in root.into_inner() {
         match stmt_pair.as_rule() {
            Rule::stmt => {
                let inner = stmt_pair.into_inner().next().unwrap();
                match inner.as_rule() {
                    Rule::func_def => stmts.push(Stmt::FuncDef(parse_func_def(inner)?)),
                    Rule::assign => stmts.push(Stmt::Assign(parse_assign(inner)?)),
                    _ => unreachable!(),
                }
            }
            Rule::EOI => break,
            _ => unreachable!(),
        }
    }
    Ok(stmts)
}

fn parse_func_def(pair: Pair<Rule>) -> Result<FuncDef> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    
    // Could be args (Case A) or type_def (Case B or Case A no args)
    let next = inner.next().unwrap();
    
    let (args, return_type) = if next.as_rule() == Rule::args {
        // Case A with args: name(args): Type
        let args = parse_args(next)?;
        let ret = parse_type_def(inner.next().unwrap())?;
        (args, ret)
    } else {
        // Case B (name: Type(args)) OR Case A/B (no args)
        // next is type_def
        let ret = parse_type_def(next)?;
        
        // check if args follow
        let mut args = HashMap::new();
        if let Some(peek_pair) = inner.peek() {
            if peek_pair.as_rule() == Rule::args {
                args = parse_args(inner.next().unwrap())?;
            }
        }
        (args, ret)
    };
    
    // body (block | expr)
    let body_pair = inner.next().unwrap();
    let body = match body_pair.as_rule() {
        Rule::block => parse_block(body_pair)?,
        Rule::expr => Block { assigns: vec![
            Assign {
                target: ("result".to_string(), None),
                expr: parse_expr(body_pair)?,
            }
        ]},
        _ => unreachable!("Unexpected rule in func_def body: {:?}", body_pair.as_rule()),
    };

    Ok(FuncDef {
        name,
        args,
        return_type,
        body,
    })
}

fn parse_args(pair: Pair<Rule>) -> Result<HashMap<String, String>> {
    let mut args = HashMap::new();
    for arg in pair.into_inner() {
        let mut inner = arg.into_inner();
        let name = inner.next().unwrap().as_str().to_string();
        let type_def = parse_type_def(inner.next().unwrap())?;
        
        // consume optional default value (op_assign ~ literal)
        if let Some(_assign) = inner.next() {
            // ignore default value for now
            let _val = inner.next(); 
        }
        
        args.insert(name, type_def);
    }
    Ok(args)
}

fn parse_type_def(pair: Pair<Rule>) -> Result<String> {
    Ok(pair.as_str().replace(" ", ""))
}

fn parse_assign(pair: Pair<Rule>) -> Result<Assign> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    
    let next = inner.next().unwrap();
    let (type_ann, expr_pair) = if next.as_rule() == Rule::type_def {
        (Some(parse_type_def(next)?), inner.next().unwrap())
    } else {
        (None, next)
    };
    
    let expr = parse_expr(expr_pair)?;
    Ok(Assign {
        target: (name, type_ann),
        expr,
    })
}

fn parse_block(pair: Pair<Rule>) -> Result<Block> {
    let mut assigns = Vec::new();
    for inner in pair.into_inner() {
         if inner.as_rule() == Rule::assign {
             assigns.push(parse_assign(inner)?);
         }
    }
    Ok(Block { assigns })
}

fn parse_expr(pair: Pair<Rule>) -> Result<Expr> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::literal => parse_literal(inner),
        Rule::call => parse_call(inner),
        Rule::var_ref => Ok(Expr::VarRef(inner.as_str().to_string())),
        Rule::block => Ok(Expr::Block(parse_block(inner)?)),
        _ => unreachable!("Unexpected rule in expr: {:?}", inner.as_rule()),
    }
}

fn parse_literal(pair: Pair<Rule>) -> Result<Expr> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::float => Ok(Expr::Literal(Literal::Float(inner.as_str().parse()?))),
        Rule::int => Ok(Expr::Literal(Literal::Int(inner.as_str().parse()?))),
        Rule::string => Ok(Expr::Literal(Literal::String(inner.as_str().trim_matches('"').to_string()))),
        Rule::bool => Ok(Expr::Literal(Literal::Bool(inner.as_str() == "True"))),
        _ => unreachable!("Unexpected rule in literal: {:?}", inner.as_rule()),
    }
}

fn parse_call(pair: Pair<Rule>) -> Result<Expr> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let mut args = HashMap::new();
    
    if let Some(call_args) = inner.next() {
        for (i, arg_pair) in call_args.into_inner().enumerate() {
            let mut arg_inner = arg_pair.into_inner();
            let first = arg_inner.next().unwrap();
            
            let (key, expr_pair) = if first.as_rule() == Rule::ident {
                let k = first.as_str().to_string();
                let e = arg_inner.next().unwrap();
                (k, e)
            } else {
                (i.to_string(), first)
            };
            args.insert(key, parse_expr(expr_pair)?);
        }
    }
    
    Ok(Expr::Call(name, args))
}
