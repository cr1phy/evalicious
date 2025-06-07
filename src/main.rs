use nom::{
    IResult, Parser, branch::alt, bytes::complete::tag, character::complete::space0,
    combinator::opt, error::ParseError, number::complete::float, sequence::delimited,
};

#[derive(Debug, PartialEq, Clone)]
enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power, // Добавлена операция "степень"
}

#[derive(Debug, PartialEq, Clone)]
enum Function {
    Sin,
    Cos,
    Tan,
    Sqrt,
    Log,   // Натуральный логарифм
    Log10, // Десятичный логарифм
    Abs,
    Floor,
    Ceil,
    Round,
    Factorial,
    // ... другие функции могут быть добавлены
}

#[derive(Debug, PartialEq, Clone)]
enum Expr {
    Number(f64),
    BinaryOp {
        op: Operation,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    FunctionCall {
        // Добавлен вариант для вызова функций
        func: Function,
        arg: Box<Expr>,
    },
}

// Парсер для чисел с плавающей точкой
fn parse_number<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Expr, E> {
    let (input, number) = delimited(
        space0, float, // Используем float парсер напрямую
        space0,
    )(input)?;
    Ok((input, Expr::Number(number.into())))
}

// Парсер для операций
fn parse_operation<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Operation, E> {
    alt((
        nom::character::complete::char('+').map(|_| Operation::Add),
        nom::character::complete::char('-').map(|_| Operation::Subtract),
        nom::character::complete::char('*').map(|_| Operation::Multiply),
        nom::character::complete::char('/').map(|_| Operation::Divide),
        nom::character::complete::char('^').map(|_| Operation::Power), // Добавлено распознавание "^"
    ))(input)
}

fn parse_function_call<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Expr, E> {
    let (input, func_name_str) = delimited(
        space0,
        alt((
            tag("sqrt"),
            tag("sin"),
            tag("cos"),
            tag("tan"),
            tag("log"),
            tag("log10"),
            tag("abs"),
            tag("floor"),
            tag("ceil"),
            tag("round"),
            tag("factorial"),
        )),
        space0,
    )(input)?;

    let func = match func_name_str {
        "sqrt" => Function::Sqrt,
        "sin" => Function::Sin,
        "cos" => Function::Cos,
        "tan" => Function::Tan,
        "log" => Function::Log,
        "log10" => Function::Log10,
        "abs" => Function::Abs,
        "floor" => Function::Floor,
        "ceil" => Function::Ceil,
        "round" => Function::Round,
        "factorial" => Function::Factorial,
        _ => {
            return Err(nom::Err::Error(E::from_error_kind(
                input,
                nom::error::ErrorKind::Tag,
            )));
        } // Неизвестная функция (ошибка)
    };

    let (input, arg) = delimited(
        delimited(space0, tag("("), space0),
        parse_expression, // Аргумент функции - это выражение
        delimited(space0, tag(")"), space0),
    )(input)?;

    Ok((input, Expr::FunctionCall {
        func,
        arg: Box::new(arg),
    }))
}

// Парсер для факторов (самый высокий приоритет - числа, функции или выражения в скобках)
fn parse_factor<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Expr, E> {
    alt((
        parse_number,
        parse_function_call, // Добавили парсер для функций
        delimited(
            delimited(space0, tag("("), space0),
            parse_expression,
            delimited(space0, tag(")"), space0),
        ),
    ))(input)
}

// Парсер для термов (умножение и деление, степень) - имеет более высокий приоритет
fn parse_term<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Expr, E> {
    let (input, left) = parse_factor(input)?; // Начинаем с факторов (числа или скобки)
    parse_term_recursive(input, left)
}

// Рекурсивный парсер для термов
fn parse_term_recursive<'a, E: ParseError<&'a str>>(
    input: &'a str,
    left_expr: Expr,
) -> IResult<&'a str, Expr, E> {
    let (input, operation) = opt(alt((
        nom::character::complete::char('*').map(|_| Operation::Multiply),
        nom::character::complete::char('/').map(|_| Operation::Divide),
        nom::character::complete::char('^').map(|_| Operation::Power), // Добавили степень в термы (можно изменить приоритет, если нужно)
    )))(input)?; // опциональная операция (*, /, ^)

    match operation {
        Some(op) => {
            let (input, right_expr) = parse_factor(input)?; // парсим следующий фактор
            let next_expr = Expr::BinaryOp {
                op,
                left: Box::new(left_expr),
                right: Box::new(right_expr),
            };
            parse_term_recursive(input, next_expr) // рекурсивно продолжаем разбор
        }
        None => Ok((input, left_expr)), // если нет операции, возвращаем текущий терм
    }
}

// Парсер для выражений (сложение и вычитание)
fn parse_expression<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, Expr, E> {
    let (input, left) = parse_term(input)?; // Начинаем с термов (умножение/деление/степень)
    parse_expression_recursive(input, left)
}

// Рекурсивный парсер для выражений (для обработки приоритета операций)
fn parse_expression_recursive<'a, E: ParseError<&'a str>>(
    input: &'a str,
    left_expr: Expr,
) -> IResult<&'a str, Expr, E> {
    let (input, operation) = opt(parse_operation)(input)?; // опциональная операция (+ или -)

    match operation {
        Some(op) => {
            let (input, right_expr) = parse_term(input)?; // парсим следующий терм
            let next_expr = Expr::BinaryOp {
                op,
                left: Box::new(left_expr),
                right: Box::new(right_expr),
            };
            parse_expression_recursive(input, next_expr) // рекурсивно продолжаем разбор
        }
        None => Ok((input, left_expr)), // если нет операции, возвращаем текущее выражение
    }
}

// Функция для вычисления выражения (интерпретация AST)
fn evaluate_expr(expr: &Expr) -> Result<f64, String> {
    match expr {
        Expr::Number(n) => Ok(*n),
        Expr::BinaryOp { op, left, right } => {
            let left_val = evaluate_expr(left)?;
            let right_val = evaluate_expr(right)?;
            match op {
                Operation::Add => Ok(left_val + right_val),
                Operation::Subtract => Ok(left_val - right_val),
                Operation::Multiply => Ok(left_val * right_val),
                Operation::Divide => {
                    if right_val == 0.0 {
                        Err("Division by zero".to_string())
                    } else {
                        Ok(left_val / right_val)
                    }
                }
                Operation::Power => Ok(left_val.powf(right_val)), // Добавлена обработка степени
            }
        }
        Expr::FunctionCall { func, arg } => {
            let arg_val = evaluate_expr(arg)?; // Вычисляем аргумент функции
            match func {
                Function::Sin => Ok(arg_val.sin()),
                Function::Cos => Ok(arg_val.cos()),
                Function::Tan => Ok(arg_val.tan()),
                Function::Sqrt => {
                    if arg_val < 0.0 {
                        Err("Square root of negative number".to_string()) // Обработка ошибки для sqrt(-1)
                    } else {
                        Ok(arg_val.sqrt())
                    }
                }
                Function::Log => Ok(arg_val.ln()), // Натуральный логарифм
                Function::Log10 => Ok(arg_val.log10()), // Десятичный логарифм
                Function::Abs => Ok(arg_val.abs()),
                Function::Floor => Ok(arg_val.floor()),
                Function::Ceil => Ok(arg_val.ceil()),
                Function::Round => Ok(arg_val.round()),
                Function::Factorial => {
                    if arg_val < 0.0 || arg_val != arg_val.floor() {
                        // Факториал только для неотрицательных целых
                        Err("Factorial is defined for non-negative integers only".to_string())
                    } else {
                        let n = arg_val as u64; // Преобразуем в u64 для факториала
                        Ok(factorial(n) as f64) // Вызываем функцию factorial
                    }
                } // ... другие функции будут добавлены сюда
            }
        }
    }
}

// Функция для вычисления факториала (простая реализация)
fn factorial(n: u64) -> u64 {
    if n == 0 { 1 } else { n * factorial(n - 1) }
}

fn main() {
    let expression = "sqrt(2) + sin(3.14159) * 2 ^ 3";

    match parse_expression::<nom::error::VerboseError<&str>>(expression) {
        Ok((remaining_input, parsed_expr)) => {
            println!("Разобранное выражение: {:?}", parsed_expr);
            match evaluate_expr(&parsed_expr) {
                Ok(result) => println!("Результат: {}", result),
                Err(error) => println!("Ошибка вычисления: {}", error),
            }
            if !remaining_input.is_empty() {
                println!("Остаток после разбора: '{}'", remaining_input);
            }
        }
        Err(e) => {
            println!("Ошибка парсинга: {:?}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{evaluate_expr, parse_expression};

    #[test]
    fn test_sqrt_function() {
        assert_eq!(evaluate_expression_str("sqrt(9)").unwrap(), 3.0);
        assert_eq!(evaluate_expression_str("sqrt(2)").unwrap(), 2.0_f64.sqrt());
        assert!(evaluate_expression_str("sqrt(-1)").is_err());
    }

    #[test]
    fn test_sin_function() {
        assert_eq!(evaluate_expression_str("sin(0)").unwrap(), 0.0);
        assert_eq!(
            evaluate_expression_str("sin(3.14159/2)").unwrap().round(),
            1.0
        );
    }

    #[test]
    fn test_factorial_function() {
        assert_eq!(evaluate_expression_str("factorial(0)").unwrap(), 1.0);
        assert_eq!(evaluate_expression_str("factorial(5)").unwrap(), 120.0);
        assert!(evaluate_expression_str("factorial(-1)").is_err());
        assert!(evaluate_expression_str("factorial(3.5)").is_err());
    }

    #[test]
    fn test_power_operation() {
        assert_eq!(evaluate_expression_str("2^3").unwrap(), 8.0);
        assert_eq!(evaluate_expression_str("2^0").unwrap(), 1.0);
        assert_eq!(evaluate_expression_str("2^-1").unwrap(), 0.5);
    }
    // Вспомогательная функция для тестирования
    fn evaluate_expression_str(expression: &str) -> Result<f64, String> {
        match parse_expression::<nom::error::VerboseError<&str>>(expression) {
            Ok((_, parsed_expr)) => evaluate_expr(&parsed_expr),
            Err(e) => Err(format!("Parsing error: {:?}", e)),
        }
    }
}
