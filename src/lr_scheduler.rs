use std::fmt::Display;

use dfdx::{optim::Adam, shapes::Dtype, tensor::DeviceStorage};

pub trait LearningRate<E> {
    fn learning_rate(&mut self) -> &mut E;
}

impl<M, E: Dtype, D: DeviceStorage> LearningRate<E> for Adam<M, E, D> {
    fn learning_rate(&mut self) -> &mut E {
        &mut self.cfg.lr
    }
}

pub trait Scheduler<E> {
    fn get(&self) -> E;
}

pub trait LRScheduler<E, T: LearningRate<E>> {
    fn step(&mut self, optimizer: &mut T);
}

impl<E: Dtype + Display, L, S> LRScheduler<E, L> for S
where
    L: LearningRate<E>,
    S: Scheduler<E>,
{
    fn step(&mut self, optimizer: &mut L) {
        *optimizer.learning_rate() = self.get();
    }
}

pub struct OneCycleScheduler<E> {
    base: E,
    max: E,
    peak: f32,
    progress: f32,
}

impl<E: Dtype> OneCycleScheduler<E> {
    pub fn new(base: E, max: E) -> Self {
        Self {
            base,
            max,
            peak: 0.2,
            progress: 0.,
        }
    }

    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress;
    }

    pub fn set_peak(self, peak: f32) -> Self {
        Self { peak, ..self }
    }
}

impl<E: Dtype> Scheduler<E> for OneCycleScheduler<E> {
    fn get(&self) -> E {
        let progress = if self.progress < self.peak {
            self.progress / self.peak
        } else {
            1.0 - ((self.progress - self.peak) / (1.0 - self.peak))
        };
        (self.max - self.base) * E::from_f32(progress).unwrap() + self.base
    }
}

pub struct LinearScheduler<E> {
    base: E,
    max: E,
    progress: f32,
}

impl<E: Dtype> LinearScheduler<E> {
    pub fn new(base: E, max: E) -> Self {
        Self {
            base,
            max,
            progress: 0.,
        }
    }

    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress;
    }
}

impl<E: Dtype> Scheduler<E> for LinearScheduler<E> {
    fn get(&self) -> E {
        (self.max - self.base) * E::from_f32(self.progress).unwrap() + self.base
    }
}
