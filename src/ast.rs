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
