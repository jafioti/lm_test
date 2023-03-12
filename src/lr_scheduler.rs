use dfdx::{optim::Adam, shapes::Dtype, tensor::DeviceStorage};

pub trait LearningRate<E> {
    fn learning_rate(&mut self) -> &mut E;
}

impl<M, E: Dtype, D: DeviceStorage> LearningRate<E> for Adam<M, E, D> {
    fn learning_rate(&mut self) -> &mut E {
        &mut self.cfg.lr
    }
}

pub trait OptimScheduler<T> {
    fn step(&mut self, optimizer: &mut T);
}

pub struct OneCycleScheduler<E> {
    base_lr: E,
    max_lr: E,
    progress: f32,
}

impl<E> OneCycleScheduler<E> {
    pub fn new(base_lr: E, max_lr: E) -> Self {
        Self {
            base_lr,
            max_lr,
            progress: 0.,
        }
    }

    pub fn set_progress(&mut self, progress: f32) {
        self.progress = progress;
    }
}

impl<E: Dtype, L: LearningRate<E>> OptimScheduler<L> for OneCycleScheduler<E> {
    fn step(&mut self, optimizer: &mut L) {
        let progress = (0.5 - (self.progress - 0.5).abs()) * 2.;
        *optimizer.learning_rate() = 
            (self.max_lr - self.base_lr)
             * E::from_f32(progress).unwrap()
             + self.base_lr;
    }
}