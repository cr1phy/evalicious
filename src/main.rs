//! Калькулятор на Rust с парсером выражений на nom 8, поддержкой переменных, функций и REPL.
//!
//! # Модули
//! - ast: абстрактное синтаксическое дерево (AST)
//! - parser: парсер выражений и присваиваний
//! - eval: вычисление выражений
//! - main: REPL-оболочка

use std::io::{self, Write};

/// AST (абстрактное синтаксическое дерево)
pub mod ast {
    use serde::{Deserialize, Serialize};

    /// Бинарные операторы
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum BinOp {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Mod,
    }

    /// Унарные операторы
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum UniOp {
        Neg,
        Factorial,
    }

    /// Поддерживаемые функции
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum Function {
        Sin,
        Cos,
        Tan,
        Asin,
        Acos,
        Atan,
        Sqrt,
        Log,
        Log10,
        Exp,
        Abs,
        Floor,
        Ceil,
        Round,
        Sum,
        Avg,
        Max,
        Min,
        Clamp,
    }

    /// Выражение
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum Expr {
        /// Число
        Number(f64),
        /// Переменная
        Var(String),
        /// Унарная операция
        Unary {
            /// Оператор
            op: UniOp,
            /// Операнд
            expr: Box<Expr>,
        },
        /// Бинарная операция
        Binary {
            /// Оператор
            op: BinOp,
            /// Левый операнд
            lhs: Box<Expr>,
            /// Правый операнд
            rhs: Box<Expr>,
        },
        /// Вызов функции
        Call {
            /// Функция
            func: Function,
            /// Аргументы
            args: Vec<Expr>,
        },
    }

    /// Оператор присваивания или выражение
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub enum Statement {
        /// Присваивание переменной
        Assign {
            /// Имя переменной
            ident: String,
            /// Выражение
            expr: Expr,
        },
        /// Просто выражение
        Expr(Expr),
    }
}

/// Парсер выражений и присваиваний
pub mod parser {
    use super::ast::*;
    use nom::{
        IResult, Parser,
        branch::alt,
        bytes::complete::{take_while, take_while1},
        character::complete::{char, multispace0},
        combinator::{map, opt, recognize},
        error::ParseError,
        multi::separated_list1,
        sequence::{delimited, pair},
    };

    /// Оборачивает парсер, чтобы игнорировать пробелы вокруг.
    ///
    /// # Параметры
    /// - `inner`: внутренний парсер
    fn ws<'a, F, O, E>(mut inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
    where
        F: Parser<&'a str, Output = O, Error = E>,
        E: ParseError<&'a str>,
    {
        move |input| {
            let (input, _) = multispace0(input)?;
            let (input, out) = inner.parse(input)?;
            let (input, _) = multispace0(input)?;
            Ok((input, out))
        }
    }

    /// Парсер идентификатора (имя переменной или функции)
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    fn ident(input: &str) -> IResult<&str, String> {
        let first = |c: char| c.is_ascii_alphabetic() || c == '_';
        let rest = |c: char| c.is_ascii_alphanumeric() || c == '_';
        map(
            |i| ws(recognize(pair(take_while1(first), take_while(rest))))(i),
            |s: &str| s.to_string(),
        )
        .parse(input)
    }

    /// Парсер числа
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    fn number(input: &str) -> IResult<&str, Expr> {
        map(ws(nom::number::complete::double), Expr::Number).parse(input)
    }

    /// Парсер переменной или константы (pi, e)
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    fn var_or_const(input: &str) -> IResult<&str, Expr> {
        map(ident, |s| match s.as_str() {
            "pi" => Expr::Number(std::f64::consts::PI),
            "e" => Expr::Number(std::f64::consts::E),
            _ => Expr::Var(s),
        })
        .parse(input)
    }

    /// Парсер выражения (с поддержкой приоритетов)
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    fn expr(input: &str) -> IResult<&str, Expr> {
        expr_bp(input, 0)
    }

    /// Парсер постфиксного факториала
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    /// - `base`: выражение слева от '!'
    fn postfix_fact(input: &str, base: Expr) -> IResult<&str, Expr> {
        let mut input = input;
        let mut lhs = base;
        while let Ok((rest, _)) = ws::<_, _, nom::error::Error<&str>>(char('!'))(input) {
            lhs = Expr::Unary {
                op: UniOp::Factorial,
                expr: Box::new(lhs),
            };
            input = rest;
        }
        Ok((input, lhs))
    }

    /// Парсер вызова функции с аргументами
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    fn func_call(input: &str) -> IResult<&str, Expr> {
        let (input, name) = ident(input)?;
        let (input, args) = delimited(
            ws(char('(')),
            separated_list1(ws(char(',')), expr),
            ws(char(')')),
        )
        .parse(input)?;
        let func = match name.as_str() {
            "sin" => Function::Sin,
            "cos" => Function::Cos,
            "tan" => Function::Tan,
            "asin" => Function::Asin,
            "acos" => Function::Acos,
            "atan" => Function::Atan,
            "sqrt" => Function::Sqrt,
            "log" => Function::Log,
            "log10" => Function::Log10,
            "exp" => Function::Exp,
            "abs" => Function::Abs,
            "floor" => Function::Floor,
            "ceil" => Function::Ceil,
            "round" => Function::Round,
            "sum" => Function::Sum,
            "avg" => Function::Avg,
            "max" => Function::Max,
            "min" => Function::Min,
            "clamp" => Function::Clamp,
            _ => {
                return Err(nom::Err::Error(nom::error::Error::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )));
            }
        };
        Ok((input, Expr::Call { func, args }))
    }

    /// Парсер атомарных выражений (число, переменная, вызов функции, скобки)
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    fn atom(input: &str) -> IResult<&str, Expr> {
        let (input, base) = alt((
            number,
            var_or_const,
            func_call,
            delimited(ws(char('(')), expr, ws(char(')'))),
        ))
        .parse(input)?;
        postfix_fact(input, base)
    }

    /// Возвращает приоритет и тип бинарного оператора
    ///
    /// # Параметры
    /// - `ch`: символ оператора
    fn binding(ch: char) -> Option<(u8, BinOp, bool)> {
        Some(match ch {
            '+' => (0, BinOp::Add, false),
            '-' => (0, BinOp::Sub, false),
            '*' => (1, BinOp::Mul, false),
            '/' => (1, BinOp::Div, false),
            '%' => (1, BinOp::Mod, false),
            '^' => (2, BinOp::Pow, true),
            _ => return None,
        })
    }

    /// Парсер выражения с учетом приоритетов (precedence climbing)
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    /// - `min_bp`: минимальный приоритет
    fn expr_bp(input: &str, min_bp: u8) -> IResult<&str, Expr> {
        let (input_after_unary, unary_opt) = opt(ws(alt((char('+'), char('-'))))).parse(input)?;
        let (mut input, mut lhs) = match unary_opt {
            Some('+') => atom(input_after_unary)?,
            Some('-') => {
                let (r, e) = atom(input_after_unary)?;
                (r, Expr::Unary {
                    op: UniOp::Neg,
                    expr: Box::new(e),
                })
            }
            _ => atom(input_after_unary)?,
        };

        while let Some(c) = input.chars().next() {
            let (bp, op, right_assoc) = match binding(c) {
                Some(t) => t,
                None => break,
            };
            if bp < min_bp {
                break;
            }
            let (rest, _) = ws(char(c))(input)?;
            let next_min = if right_assoc { bp } else { bp + 1 };
            let (rest, rhs) = expr_bp(rest, next_min)?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
            input = rest;
        }
        Ok((input, lhs))
    }

    /// Парсер строки как присваивания или выражения
    ///
    /// # Параметры
    /// - `input`: строка для парсинга
    pub fn statement(input: &str) -> IResult<&str, Statement> {
        if input.contains('=') {
            let (input, id) = ident(input)?;
            let (input, _) = ws(char('='))(input)?;
            let (input, rhs) = expr(input)?;
            Ok((input, Statement::Assign {
                ident: id,
                expr: rhs,
            }))
        } else {
            let (input, e) = nom::combinator::all_consuming(expr).parse(input)?;
            Ok((input, Statement::Expr(e)))
        }
    }
}

/// Модуль вычисления выражений
pub mod eval {
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
}

/// REPL-оболочка калькулятора
fn main() {
    let mut ctx = eval::Context::new();
    println!(
        "Rust Calc v2 (nom 8). 'exit' для выхода, 'deg'/'rad' — переключение градусов/радиан."
    );
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let line = input.trim();
        if line.eq_ignore_ascii_case("exit") {
            break;
        }
        if line.eq_ignore_ascii_case("deg") {
            ctx.angle_in_deg = true;
            println!("→ градусов");
            continue;
        }
        if line.eq_ignore_ascii_case("rad") {
            ctx.angle_in_deg = false;
            println!("→ радианы");
            continue;
        }
        match parser::statement(line) {
            Ok((rest, stmt)) if rest.trim().is_empty() => match eval::eval_stmt(&stmt, &mut ctx) {
                Ok(v) => println!("= {v}"),
                Err(e) => eprintln!("err: {e}"),
            },
            Ok((_rest, _)) => eprintln!("Неразобранный хвост"),
            Err(e) => eprintln!("Parse error: {:?}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Выполнить выражение в новом контексте
    ///
    /// # Параметры
    /// - `s`: строка выражения
    fn eval_once(s: &str) -> f64 {
        let mut ctx = eval::Context::new();
        let (_, st) = parser::statement(s).unwrap();
        eval::eval_stmt(&st, &mut ctx).unwrap()
    }
    #[test]
    fn basic() {
        assert_eq!(eval_once("2+3*4"), 14.0);
        assert_eq!(eval_once("(2+3)*4"), 20.0);
        assert_eq!(eval_once("2^3^2"), 512.0);
    }
}
