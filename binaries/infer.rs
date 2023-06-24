use dataflow_nlp::{
    tokenization::{Tokenizer, WordpieceTokenizer},
    vocab::{Vocab, WordPieceVocab},
};
use dfdx::prelude::*;

use lm_test::{
    model::{BuiltModel, Model},
    utils::*,
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use rand::{distributions::WeightedIndex, thread_rng};
use rand_distr::Distribution;
use std::{error::Error, io, time::Duration};
use tui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Span, Spans, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use unicode_width::UnicodeWidthStr;

enum InputMode {
    Normal,
    Editing,
}

#[derive(Clone, Debug)]
enum Message {
    Assistant(String),
    User(String),
}

/// App holds the state of the application
struct App<
    const V: usize,
    const E: usize,
    const F: usize,
    const L: usize,
    const H: usize,
    const M: usize,
    D: Device<f32>,
> {
    /// Current value of the input box
    input: String,
    /// Current input mode
    input_mode: InputMode,
    /// History of recorded messages
    messages: Vec<Message>,
    model: BuiltModel<V, E, F, L, H, M, f32, D>,
}

const LAYERS: usize = 8;
const MAX_SEQ_LEN: usize = 512;
const EMBED_DIM: usize = 512;
const FF_DIM: usize = EMBED_DIM * 4;
const HEADS: usize = 8;

const MAX_TRAIN_SEQ_LEN: usize = 45;

fn main() -> Result<(), Box<dyn Error>> {
    let dev = Cpu::default();
    let mut model =
        Model::<30528, EMBED_DIM, FF_DIM, LAYERS, HEADS, MAX_SEQ_LEN>::build_on_device(&dev);
    model.load("../checkpoints/new_best.npz").unwrap();

    let param_msg = format!(
        "Model Parameters: {}",
        pretty_print_num(model.num_trainable_params())
    );

    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // create app and run it
    let app = App {
        input: "".to_string(),
        input_mode: InputMode::Normal,
        messages: vec![Message::Assistant(param_msg)],
        model,
    };
    let res = run_app(&mut terminal, app, &dev);

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

fn run_app<
    B: Backend,
    const V: usize,
    const E: usize,
    const F: usize,
    const L: usize,
    const H: usize,
    const M: usize,
    D: Device<f32>,
>(
    terminal: &mut Terminal<B>,
    mut app: App<V, E, F, L, H, M, D>,
    dev: &D,
) -> io::Result<()> {
    let mut process: Option<std::thread::JoinHandle<String>> = None;

    loop {
        if let Some(p) = process.take() {
            if p.is_finished() {
                app.messages.push(Message::Assistant(p.join().unwrap()));
            } else {
                process = Some(p);
            }
        }

        terminal.draw(|f| ui(f, &app))?;

        if event::poll(Duration::from_millis(20))? {
            if let Event::Key(key) = event::read()? {
                match app.input_mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('e') => {
                            app.input_mode = InputMode::Editing;
                        }
                        KeyCode::Char('q') | KeyCode::Esc => {
                            return Ok(());
                        }
                        _ => {}
                    },
                    InputMode::Editing => match key.code {
                        KeyCode::Enter => {
                            let string: String = app.input.drain(..).collect();
                            app.messages.push(Message::User(string.clone()));
                            app.messages.push(Message::Assistant(generate(
                                &string,
                                &app.model,
                                dev,
                                50,
                                MAX_TRAIN_SEQ_LEN,
                                0.8,
                            )))
                        }
                        KeyCode::Char(c) => {
                            app.input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.input.pop();
                        }
                        KeyCode::Esc => {
                            app.input_mode = InputMode::Normal;
                        }
                        _ => {}
                    },
                }
            }
        }
    }
}

fn ui<
    B: Backend,
    const V: usize,
    const E: usize,
    const F: usize,
    const L: usize,
    const H: usize,
    const M: usize,
    D: Device<f32>,
>(
    f: &mut Frame<B>,
    app: &App<V, E, F, L, H, M, D>,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints(
            [
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(3),
            ]
            .as_ref(),
        )
        .split(f.size());

    let (msg, style) = match app.input_mode {
        InputMode::Normal => (
            vec![
                Span::raw("Press "),
                Span::styled("q / Esc", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" to exit, "),
                Span::styled("e", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" to start editing."),
            ],
            Style::default(),
        ),
        InputMode::Editing => (
            vec![
                Span::raw("Press "),
                Span::styled("Esc", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" to stop editing, "),
                Span::styled("Enter", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(" to record the message"),
            ],
            Style::default(),
        ),
    };
    let mut text = Text::from(Spans::from(msg));
    text.patch_style(style);
    let help_message = Paragraph::new(text);
    f.render_widget(help_message, chunks[0]);

    let input = Paragraph::new(app.input.as_ref())
        .style(match app.input_mode {
            InputMode::Normal => Style::default(),
            InputMode::Editing => Style::default().fg(Color::LightGreen),
        })
        .block(Block::default().borders(Borders::ALL).title("Input"));
    match app.input_mode {
        InputMode::Normal =>
            // Hide the cursor. `Frame` does this by default, so we don't need to do anything here
            {}

        InputMode::Editing => {
            // Make the cursor visible and ask tui-rs to put it at the specified coordinates after rendering
            f.set_cursor(
                // Put cursor past the end of the input text
                chunks[2].x + app.input.width() as u16 + 1,
                // Move one line down, from the border to the input line
                chunks[2].y + 1,
            )
        }
    }

    let messages: Vec<Spans> = app
        .messages
        .iter()
        .map(|m| match m {
            Message::User(m) => Spans::from(Span::raw(format!("{m}\n"))),
            Message::Assistant(m) => Spans::from(Span::styled(
                format!("{m}\n"),
                Style::default().fg(Color::LightBlue),
            )),
        })
        .collect();
    let messages = Paragraph::new(messages)
        .block(Block::default().borders(Borders::ALL).title("Messages"))
        .wrap(Wrap { trim: true });
    f.render_widget(messages, chunks[1]);
    f.render_widget(input, chunks[2]);
}

fn generate<
    const LAYERS: usize,
    const VOCAB: usize,
    const EMBED: usize,
    const FF: usize,
    const HEADS: usize,
    const MAX_LEN: usize,
    D: Device<f32>,
>(
    input: &str,
    model: &BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, MAX_LEN, f32, D>,
    dev: &D,
    num_tokens: usize,
    window_size: usize,
    temperature: f32,
) -> String
where
    BuiltModel<VOCAB, EMBED, FF, LAYERS, HEADS, MAX_LEN, f32, D>:
        Module<Tensor<(usize,), usize, D>, Output = Tensor<(usize, Const<VOCAB>), f32, D>>,
    D: Device<f32>,
{
    let (tokenizer, vocab) = (
        <WordpieceTokenizer as Tokenizer>::load(),
        <WordPieceVocab as Vocab>::load(),
    );
    let tokens = tokenizer.tokenize(&input.to_lowercase());
    let mut indexes = vocab.indexes_from_tokens(&tokens).unwrap();
    let initial_len = indexes.len();
    let mut rng = thread_rng();

    for _ in 0..num_tokens {
        let output = model.forward(dev.tensor_from_vec(
            indexes[indexes.len().checked_sub(window_size).unwrap_or_default()..].to_vec(),
            (indexes.len().min(window_size),),
        ));
        let mut distr: Vec<f32> =
            output.as_vec()[(indexes.len() - 1).min(window_size - 1) * VOCAB..].to_vec();
        softmax(&mut distr, temperature);
        indexes.push(WeightedIndex::new(&distr).unwrap().sample(&mut rng));
    }
    tokenizer.untokenize(vocab.tokens_from_indexes(&indexes[initial_len..]).unwrap())
}

fn softmax(distr: &mut [f32], temperature: f32) {
    let sum: f32 = distr.iter().map(|i| (i / temperature).exp()).sum();
    for i in distr {
        *i = (*i / temperature).exp() / sum;
    }
}
