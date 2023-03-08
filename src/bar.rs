use std::collections::HashMap;

use crate::indicatif::{ProgressBar, ProgressStyle};
use num::{Float, Zero};
use tensorboard_rs::summary_writer::SummaryWriter;
use rand::Rng;

/// Creates a training stylized progress bar
pub fn train_progress_bar(steps: u64) -> ProgressBar {
    let bar = ProgressBar::new(steps);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green.bright} {percent}│{wide_bar:.green.bright/blue}│{pos:>7}/{len:7}({msg} | {rate} | {eta} | {elapsed_precise})")
        .with_key("eta", |state| {
            let secs = state.eta().as_secs();
            let mut string = format!("{:.1}s", state.eta().as_secs_f64() % 60.);
            if (secs / 60) % 60 > 0 {string = format!("{}m {}", (secs / 60) % 60, string);}
            if (secs / 3600) % 24 > 0 {string = format!("{}h {}", (secs / 3600) % 24, string);}
            if secs / 86400 > 0 {string = format!("{}d {}", secs / 86400, string);}
            string
        })
        .with_key("rate", |state| {
            if state.per_sec() < 1. {
                format!("{:.1}s/it", 1. / state.per_sec())
            } else {
                format!("{:.1}it/s", state.per_sec())
            }
        })
        .with_key("percent", |state| {
            format!("{:.1}%", (state.pos as f32 / state.len as f32) * 100.)
        })
        .progress_chars("█▉▊▋▌▍▎▏  "));
    bar
}

/// Creates a test stylized progress bar
pub fn test_progress_bar(steps: u64) -> ProgressBar {
    let bar = ProgressBar::new(steps);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.yellow.bright} {percent}│{wide_bar:.yellow.bright/blue}│{pos:>7}/{len:7}({msg} | {rate} | {eta} | {elapsed_precise})")
        .with_key("eta", |state| {
            let secs = state.eta().as_secs();
            let mut string = format!("{:.1}s", state.eta().as_secs_f64() % 60.);
            if (secs / 60) % 60 > 0 {string = format!("{}m {}", (secs / 60) % 60, string);}
            if (secs / 3600) % 24 > 0 {string = format!("{}h {}", (secs / 3600) % 24, string);}
            if secs / 86400 > 0 {string = format!("{}d {}", secs / 86400, string);}
            string
        })
        .with_key("rate", |state| {
            if state.per_sec() < 1. {
                format!("{:.1}s/it", 1. / state.per_sec())
            } else {
                format!("{:.1}it/s", state.per_sec())
            }
        })
        .with_key("percent", |state| {
            format!("{:.1}%", (state.pos as f32 / state.len as f32) * 100.)
        })
        .progress_chars("█▉▊▋▌▍▎▏  "));
    bar
}

pub struct ExponentialAverage<T: Float> {
    beta: f64,
    moment: f64,
    pub value: T,
    t: i32
}

impl <T: Float> Default for ExponentialAverage<T> {
    fn default() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }
}

impl <T: Float> ExponentialAverage<T> {
    pub fn new() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }

    pub fn with_beta(beta: f64) -> Self {
        assert!((0. ..=1.).contains(&beta));
        ExponentialAverage {
            beta,
            moment: 0.,
            value: Zero::zero(),
            t: 0
        }
    }

    pub fn update(&mut self, value: T) {
        self.t += 1;
        self.moment = self.beta * self.moment + (1. - self.beta) * value.to_f64().unwrap();
        // bias correction
        self.value = T::from(self.moment / (1. - f64::powi(self.beta, self.t))).unwrap();
    }

    pub fn reset(&mut self) {
        self.moment = 0.;
        self.value = Zero::zero();
        self.t = 0;
    }
}

pub fn pretty_print_num(num: usize) -> String {
    match num {
        1_000_000_000_000.. => format!("{:.2}T", num as f64 / 1_000_000_000_000.),
        1_000_000_000..=999_999_999_999 => format!("{:.2}B", num as f64 / 1_000_000_000.),
        1_000_000..=999_999_999 => format!("{:.2}M", num as f64 / 1_000_000.),
        1_000..=999_999 => format!("{:.2}K", num as f64 / 1_000.),
        _ => num.to_string()
    }
}

pub struct Tensorboard {
    writer: SummaryWriter,
    pub iter: usize,
    id: String,
}

impl Tensorboard {
    pub fn new(logdir: &str) -> Self {
        Self {
            writer: tensorboard_rs::summary_writer::SummaryWriter::new(logdir),
            id: rand::thread_rng()
                .sample_iter(&rand::distributions::Alphanumeric)
                .take(20)
                .map(char::from)
                .collect::<String>(),
            iter: 0,
        }
    }

    pub fn record(&mut self, metric_name: &str, metric: f32, steps: usize) {
        self.iter += steps;
        self.writer.add_scalars(metric_name, &HashMap::from([(self.id.clone(), metric)]), self.iter);
    }
}