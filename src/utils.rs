#![allow(unused, clippy::type_complexity)]

use std::collections::HashMap;

use crate::indicatif::{ProgressBar, ProgressStyle};
use num::{Float, Zero};
use rand::Rng;
use tensorboard_rs::summary_writer::SummaryWriter;

use dfdx::{optim::Adam, prelude::*};
use half::f16;

use std::fmt::Display;

pub fn vec_one_hot_encode<const V: usize, E: Dtype + Float, D: Device<E>>(
    labels: &Vec<Vec<usize>>,
    dev: &D,
) -> Result<Tensor<(usize, usize, Const<V>), E, D>, D::Err> {
    let (batch_size, seq_len) = (labels.len(), labels[0].len());
    let mut data = vec![E::zero(); batch_size * seq_len * V];
    for (b, batch) in labels.iter().enumerate() {
        for (i, l) in batch.iter().enumerate() {
            data[b * seq_len * V + i * V + *l] = E::ONE;
        }
    }
    dev.try_tensor_from_vec(data, (batch_size, seq_len, Const::<V>))
}

pub fn accuracy<const V: usize, B: Dim, S: Dim, E: Dtype + Float, D: Device<E>>(
    output: &Tensor<(B, S, Const<V>), E, D>,
    labels: &Vec<Vec<usize>>,
) -> f32 {
    let (batch_size, seq_len) = (labels.len(), labels[0].len());
    let total_num = labels.iter().map(|i| i.len()).sum::<usize>();
    let out_vec = output.as_vec();
    let num_right = labels
        .iter()
        .enumerate()
        .map(|(seq_num, seq)| {
            seq.iter()
                .enumerate()
                .filter(|(elem_num, elem)| {
                    out_vec[(seq_num * seq_len * V + elem_num * V)
                        ..(seq_num * seq_len * V + (elem_num + 1) * V)]
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(index, _)| index)
                        .unwrap()
                        == **elem
                })
                .count()
        })
        .sum::<usize>();

    num_right as f32 / total_num as f32
}

pub fn try_cross_entropy_with_logits_loss<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>>(
    logits: Tensor<S, E, D, T>,
    target_probs: Tensor<S, E, D>,
) -> Result<Tensor<Rank0, E, D, T>, D::Err> {
    let last_axis_numel = E::from_usize(<S as HasAxes<S::LastAxis>>::size(logits.shape())).unwrap();
    logits
        .try_log_softmax::<S::LastAxis>()?
        .try_mul(target_probs)?
        .try_mean()?
        .try_negate()?
        .try_mul(last_axis_numel)
}

pub fn pretty_print_num(num: usize) -> String {
    match num {
        1_000_000_000_000.. => format!("{:.2}T", num as f64 / 1_000_000_000_000.),
        1_000_000_000..=999_999_999_999 => format!("{:.2}B", num as f64 / 1_000_000_000.),
        1_000_000..=999_999_999 => format!("{:.2}M", num as f64 / 1_000_000.),
        1_000..=999_999 => format!("{:.2}K", num as f64 / 1_000.),
        _ => num.to_string(),
    }
}

pub struct ExponentialAverage<T: Float> {
    beta: f64,
    moment: f64,
    pub value: T,
    t: i32,
}

impl<T: Float> Default for ExponentialAverage<T> {
    fn default() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0,
        }
    }
}

impl<T: Float> ExponentialAverage<T> {
    pub fn new() -> Self {
        ExponentialAverage {
            beta: 0.99,
            moment: 0.,
            value: Zero::zero(),
            t: 0,
        }
    }

    pub fn with_beta(beta: f64) -> Self {
        assert!((0. ..=1.).contains(&beta));
        ExponentialAverage {
            beta,
            moment: 0.,
            value: Zero::zero(),
            t: 0,
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
        self.writer.add_scalars(
            metric_name,
            &HashMap::from([(self.id.clone(), metric)]),
            self.iter,
        );
    }
}

pub trait LearningRate {
    fn learning_rate(&mut self) -> &mut f64;
}

impl<M, E: Dtype, D: Storage<E>> LearningRate for Adam<M, E, D> {
    fn learning_rate(&mut self) -> &mut f64 {
        &mut self.cfg.lr
    }
}

pub trait Scheduler<E> {
    fn get(&self) -> E;

    fn step(&mut self, steps: usize);
}

pub struct OneCycleScheduler<E> {
    base: E,
    max: E,
    peak: f32,
    max_steps: usize,
    current_step: usize,
}

impl<E: Dtype> OneCycleScheduler<E> {
    pub fn new(base: E, max: E, max_steps: usize) -> Self {
        Self {
            base,
            max,
            max_steps,
            current_step: 0,
            peak: 0.2,
        }
    }

    pub fn set_peak(self, peak: f32) -> Self {
        Self { peak, ..self }
    }
}

impl<E: Dtype> Scheduler<E> for OneCycleScheduler<E> {
    fn get(&self) -> E {
        let progress = (self.current_step as f32 / self.max_steps as f32)
            .max(0.)
            .min(1.);
        let progress = if progress < self.peak {
            progress / self.peak
        } else {
            1.0 - ((progress - self.peak) / (1.0 - self.peak))
        };
        (self.max - self.base) * E::from_f32(progress).unwrap() + self.base
    }

    fn step(&mut self, steps: usize) {
        self.current_step += steps;
    }
}

pub struct LinearScheduler<E> {
    base: E,
    max: E,
    total_steps: usize,
    progress: usize,
}

impl<E: Dtype> LinearScheduler<E> {
    pub fn new(base: E, max: E, total_steps: usize) -> Self {
        Self {
            base,
            max,
            total_steps,
            progress: 0,
        }
    }
}

impl<E: Dtype> Scheduler<E> for LinearScheduler<E> {
    fn get(&self) -> E {
        (self.max - self.base)
            * E::from_f32(self.progress as f32 / self.total_steps as f32).unwrap()
            + self.base
    }

    fn step(&mut self, steps: usize) {
        self.progress = (self.progress + steps).min(self.total_steps);
    }
}

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

pub fn softmax(distr: &mut [f32], temperature: f32) {
    let sum: f32 = distr.iter().map(|i| (i / temperature).exp()).sum();
    for i in distr {
        *i = (*i / temperature).exp() / sum;
    }
}
