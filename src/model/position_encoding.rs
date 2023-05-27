use std::fmt::Debug;

use dfdx::prelude::tensor_collection::{ModuleVisitor, TensorCollection, TensorOptions};
use dfdx::prelude::*;
use rand_distr::Uniform;

pub mod builder {
    #[derive(Debug)]
    pub struct LearnedPositionalEmbedding<const MAX_LEN: usize, const DIM: usize>;
}

impl<const V: usize, const M: usize, E: Dtype, D: Device<E>> BuildOnDevice<D, E>
    for builder::LearnedPositionalEmbedding<V, M>
where
    LearnedPositionalEmbedding<V, M, E, D>: BuildModule<D, E>,
{
    type Built = LearnedPositionalEmbedding<V, M, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

#[derive(Debug, Clone)]
pub struct LearnedPositionalEmbedding<
    const MAX_LEN: usize,
    const DIM: usize,
    E: Dtype,
    D: Storage<E>,
> {
    /// Learned positonal embeddings
    pub weight: Tensor<Rank2<MAX_LEN, DIM>, E, D>,
}

/// Pass in an unbatched pre-embedded sequence, add positional embeddings in
impl<const MAX_LEN: usize, const DIM: usize, SEQ: Dim, D: Device<f32>, T: Tape<f32, D> + Debug>
    Module<Tensor<(SEQ, Const<DIM>), f32, D, T>>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
{
    type Output = Tensor<(SEQ, Const<DIM>), f32, D, T>;
    type Error = D::Err;
    fn try_forward(
        &self,
        input: Tensor<(SEQ, Const<DIM>), f32, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        self.weight
            .clone()
            .put_tape(tape)
            .try_slice((..input.shape().0.size(), ..))?
            .realize::<(SEQ, Const<DIM>)>()
            .try_add(input)
    }
}

impl<
        const MAX_LEN: usize,
        const DIM: usize,
        BATCH: Dim,
        SEQ: Dim,
        D: Device<f32>,
        T: Tape<f32, D> + Debug,
    > Module<Tensor<(BATCH, SEQ, Const<DIM>), f32, D, T>>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
{
    type Output = Tensor<(BATCH, SEQ, Const<DIM>), f32, D, T>;
    type Error = D::Err;
    fn try_forward(
        &self,
        input: Tensor<(BATCH, SEQ, Const<DIM>), f32, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        self.weight
            .clone()
            .put_tape(tape)
            .try_slice((..input.shape().1.size(), ..))?
            .realize::<(SEQ, Const<DIM>)>()
            .try_broadcast_like(input.shape())?
            .try_add(input)
    }
}

impl<const V: usize, const M: usize, E: Dtype, D: Storage<E>> NonMutableModule
    for LearnedPositionalEmbedding<V, M, E, D>
{
}

impl<const C: usize, const M: usize, D: SampleTensor<f32> + Device<f32>> TensorCollection<f32, D>
    for LearnedPositionalEmbedding<C, M, f32, D>
{
    type To<E2: Dtype, D2: Device<E2>> = LearnedPositionalEmbedding<C, M, E2, D2>;
    fn iter_tensors<V: ModuleVisitor<Self, f32, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            Self::tensor(
                "weight",
                |s| &s.weight,
                |s| &mut s.weight,
                TensorOptions::reset_with(|t| {
                    let b = 1. / (C as f32).sqrt();
                    t.try_fill_with_distr(Uniform::new(-b, b))
                }),
            ),
            |weight| LearnedPositionalEmbedding { weight },
        )
    }
}
