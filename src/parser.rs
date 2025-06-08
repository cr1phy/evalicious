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
