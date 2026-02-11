use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Any,
    Atomic(String),
    Generic(String, Box<Type>),
    Either(Vec<Type>),
    Dict(HashMap<String, Type>),
    Tuple(Vec<Type>),
    Operator(Box<Type>, Box<Type>), // Args, Return
}

impl Type {
    pub fn matches(&self, other: &Type) -> bool {
        match (self, other) {
            (Type::Any, _) => true,
            (_, Type::Any) => true, // Assuming Any matches anything in both directions for now? Or just strictly one way? Python: AnyType.matches(other) = True.
            
            (Type::Atomic(n1), Type::Atomic(n2)) => {
                let matches = n1 == n2;
                if !matches && n1 == "Int" { // Debug specific case
                     println!("DEBUG: Atomic check '{}' == '{}' -> {}", n1, n2, matches);
                     println!("DEBUG: n1 bytes: {:?}, n2 bytes: {:?}", n1.as_bytes(), n2.as_bytes());
                }
                matches
            },
            
            (Type::Generic(b1, i1), Type::Generic(b2, i2)) => {
                b1 == b2 && i1.matches(i2)
            }
            
            (Type::Either(opts), other) => {
                opts.iter().any(|opt| opt.matches(other))
            },
            
            // Note: If 'other' is Either, do we check if 'self' matches *any* of 'other's options?
            // Or does 'other' have to be a specific instance that matches 'self' (pattern)?
            // In Python: DeclType.matches(RhsType).
            // If Decl is Either[A, B], and Rhs is A, Returns True.
            // If Decl is A, and Rhs is Either[A, B], Returns False (usually, unless we narrow).
            // Python impl: Either.matches(other) = any(opt.matches(other) ... )
            
            (Type::Dict(f1), Type::Dict(f2)) => {
                // Check if all required fields in self exist in other and match
                for (k, t) in f1 {
                    match f2.get(k) {
                        Some(other_t) => {
                            if !t.matches(other_t) {
                                return false;
                            }
                        }
                        None => return false,
                    }
                }
                true
            }
            
            (Type::Tuple(i1), Type::Tuple(i2)) => {
                if i1.len() != i2.len() {
                    return false;
                }
                i1.iter().zip(i2.iter()).all(|(t1, t2)| t1.matches(t2))
            }
            
            (Type::Operator(a1, r1), Type::Operator(_a2, r2)) => {
                // Return type covariance/match
                r1.matches(r2) 
            }
            
            _ => false,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Any => write!(f, "Any"),
            Type::Atomic(n) => write!(f, "{}", n),
            Type::Generic(b, i) => write!(f, "{}<{}>", b, i),
            Type::Either(opts) => {
                let s: Vec<String> = opts.iter().map(|t| t.to_string()).collect();
                write!(f, "Either([{}])", s.join(", "))
            },
            Type::Dict(fields) => {
                let s: Vec<String> = fields.iter().map(|(k,v)| format!("{}: {}", k, v)).collect();
                write!(f, "Dict({{{}}})", s.join(", "))
            },
            Type::Tuple(items) => {
                let s: Vec<String> = items.iter().map(|t| t.to_string()).collect();
                write!(f, "Tuple(({}))", s.join(", "))
            },
            Type::Operator(a, r) => write!(f, "Operator<{}, {}>", a, r),
        }
    }
}
