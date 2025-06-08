use super::ast::*;
use std::collections::HashMap;

/// Ошибки вычисления
#[derive(Debug)]
pub enum Error {
    DivZero,
    NegativeSqrt,
    InvalidFactorial,
    Domain(String),
    UndefinedVar(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for Error {}

/// Контекст вычисления (переменные, режим углов)
#[derive(Clone)]
pub struct Context {
    /// Значения переменных
    pub vars: HashMap<String, f64>,
    /// true — градусы, false — радианы
    pub angle_in_deg: bool,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Создать новый контекст
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            angle_in_deg: false,
        }
    }
    /// Преобразовать угол в радианы, если нужно
    ///
    /// # Параметры
    /// - `x`: угол
    fn ang(&self, x: f64) -> f64 {
        if self.angle_in_deg { x.to_radians() } else { x }
    }
}

/// Вычислить оператор присваивания или выражение
///
/// # Параметры
/// - `stmt`: оператор
/// - `ctx`: контекст
pub fn eval_stmt(stmt: &Statement, ctx: &mut Context) -> Result<f64, Error> {
    match stmt {
        Statement::Assign { ident, expr } => {
            let v = eval_expr(expr, ctx)?;
            ctx.vars.insert(ident.clone(), v);
            Ok(v)
        }
        Statement::Expr(e) => eval_expr(e, ctx),
    }
}

/// Вычислить выражение
///
/// # Параметры
/// - `expr`: выражение
/// - `ctx`: контекст
pub fn eval_expr(expr: &Expr, ctx: &Context) -> Result<f64, Error> {
    use BinOp::*;
    Ok(match expr {
        Expr::Number(n) => *n,
        Expr::Var(name) => *ctx
            .vars
            .get(name)
            .ok_or_else(|| Error::UndefinedVar(name.clone()))?,
        Expr::Unary { op, expr } => {
            let v = eval_expr(expr, ctx)?;
            match op {
                UniOp::Neg => -v,
                UniOp::Factorial => fact(v)?,
            }
        }
        Expr::Binary { op, lhs, rhs } => {
            let l = eval_expr(lhs, ctx)?;
            let r = eval_expr(rhs, ctx)?;
            match op {
                Add => l + r,
                Sub => l - r,
                Mul => l * r,
                Div => {
                    if r == 0.0 {
                        return Err(Error::DivZero);
                    } else {
                        l / r
                    }
                }
                Pow => l.powf(r),
                Mod => l % r,
            }
        }
        Expr::Call { func, args } => eval_func(func, args, ctx)?,
    })
}

/// Факториал (только для неотрицательных целых)
///
/// # Параметры
/// - `x`: значение
fn fact(x: f64) -> Result<f64, Error> {
    if x < 0.0 || x.fract() != 0.0 {
        return Err(Error::InvalidFactorial);
    }
    let mut acc = 1u128;
    for i in 2..=(x as u64) {
        acc *= i as u128;
    }
    Ok(acc as f64)
}

/// Вычислить функцию
///
/// # Параметры
/// - `func`: функция
/// - `args`: аргументы
/// - `ctx`: контекст
fn eval_func(func: &Function, args: &[Expr], ctx: &Context) -> Result<f64, Error> {
    let get = |i: usize| eval_expr(&args[i], ctx);
    Ok(match func {
        Function::Sin => ctx.ang(get(0)?).sin(),
        Function::Cos => ctx.ang(get(0)?).cos(),
        Function::Tan => ctx.ang(get(0)?).tan(),
        Function::Asin => get(0)?.asin(),
        Function::Acos => get(0)?.acos(),
        Function::Atan => get(0)?.atan(),
        Function::Sqrt => {
            let v = get(0)?;
            if v < 0.0 {
                return Err(Error::NegativeSqrt);
            } else {
                v.sqrt()
            }
        }
        Function::Log => get(0)?.ln(),
        Function::Log10 => get(0)?.log10(),
        Function::Exp => get(0)?.exp(),
        Function::Abs => get(0)?.abs(),
        Function::Floor => get(0)?.floor(),
        Function::Ceil => get(0)?.ceil(),
        Function::Round => get(0)?.round(),
        Function::Sum => args
            .iter()
            .map(|e| eval_expr(e, ctx))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum(),
        Function::Avg => {
            let v = args
                .iter()
                .map(|e| eval_expr(e, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            v.iter().sum::<f64>() / v.len() as f64
        }
        Function::Max => args
            .iter()
            .map(|e| eval_expr(e, ctx))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .fold(f64::NAN, f64::max),
        Function::Min => args
            .iter()
            .map(|e| eval_expr(e, ctx))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .fold(f64::NAN, f64::min),
        Function::Clamp => {
            if args.len() != 3 {
                return Err(Error::Domain("clamp needs 3 args".into()));
            }
            let v = get(0)?;
            let lo = get(1)?;
            let hi = get(2)?;
            v.clamp(lo, hi)
        }
    })
}
