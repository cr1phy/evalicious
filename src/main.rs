use teloxide::utils::{html::italic, markdown::{bold, escape, escape_code}};

mod ast;
mod eval;
mod parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use dotenvy::dotenv;
    use teloxide::{
        prelude::*,
        types::{
            InlineQueryResult, InlineQueryResultArticle, InputMessageContent,
            InputMessageContentText, ParseMode,
        },
    };

    dotenv().ok();
    tracing_subscriber::fmt().init();
    log::info!("Starting evalicious bot…");

    let bot = Bot::from_env();

    let handler = Update::filter_inline_query().branch(dptree::endpoint(
        |bot: Bot, q: InlineQuery| async move {
            let expr = &q.query;
            let result = match parser::statement(expr) {
                Ok((rest, stmt)) if rest.trim().is_empty() => {
                    let mut ctx = eval::Context::new();
                    match eval::eval_stmt(&stmt, &mut ctx) {
                        Ok(val) => {
                            let val_fmt = format!("{val:.8}")
                                .trim_end_matches('0')
                                .trim_end_matches('.')
                                .to_string();
                            format!("🧮 Пример:\n{}\n\n📏 Ответ:\n{}", escape_code(&expr), escape_code(&val_fmt))
                        }
                        Err(e) => {
                            format!(
                                "❌ Не удалось вычислить пример:\n{}\nПричина: {}",
                                escape_code(&expr), escape_code(&e.to_string())
                            )
                        }
                    }
                }
                Ok(_) => "❌ Не удалось разобрать пример: лишний текст после выражения".to_string(),
                Err(e) => format!("❌ Ошибка парсинга: _{}_", italic(&e.to_string())),
            };

            let solve = InlineQueryResultArticle::new(
                q.id.clone(),
                "Evalicious Result",
                InputMessageContent::Text(
                    InputMessageContentText::new(result).parse_mode(ParseMode::MarkdownV2),
                ),
            )
            .description("Evaluates mathematical expressions.");

            let results = vec![InlineQueryResult::Article(solve)];

            let response = bot.answer_inline_query(&q.id, results).send().await;
            if let Err(e) = response {
                log::error!("Failed to answer inline query: {}", e);
            }
            log::info!("Received inline query: {:?}", q);
            respond(())
        },
    ));

    Dispatcher::builder(bot, handler)
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_functions() {
        assert_eq!(eval_once("sqrt(4)"), 2.0);
        assert!((eval_once("sin(pi/2)") - 1.0).abs() < 1e-8);
        assert_eq!(eval_once("log(e)"), 1.0);
        assert_eq!(eval_once("abs(-5)"), 5.0);
    }

    #[test]
    fn test_vars() {
        let mut ctx = eval::Context::new();
        let (_, st1) = parser::statement("x=3").unwrap();
        eval::eval_stmt(&st1, &mut ctx).unwrap();
        let (_, st2) = parser::statement("x^2+1").unwrap();
        assert_eq!(eval::eval_stmt(&st2, &mut ctx).unwrap(), 10.0);
    }

    #[test]
    fn test_sums_and_clamp() {
        assert_eq!(eval_once("sum(1,2,3,4)"), 10.0);
        assert_eq!(eval_once("clamp(7, 1, 5)"), 5.0);
        assert_eq!(eval_once("clamp(0, 1, 5)"), 1.0);
    }
}
