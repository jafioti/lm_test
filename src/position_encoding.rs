use dfdx::prelude::tensor_collection::{TensorCollection, ModuleVisitor, TensorOptions};
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
    D: DeviceStorage,
> {
    /// Learned positonal embeddings
    pub weight: Tensor<Rank2<MAX_LEN, DIM>, E, D>,
}

/// Pass in an unbatched pre-embedded sequence, add positional embeddings in
impl<const MAX_LEN: usize, const DIM: usize, D: Device<f32>, T: Tape<f32, D>>
    Module<Tensor<Rank2<MAX_LEN, DIM>, f32, D, T>>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
where
    D: TensorFromVec<usize>,
{
    type Output = Tensor<Rank2<MAX_LEN, DIM>, f32, D, T>;
    type Error = D::Err;
    fn try_forward(&self, input: Tensor<Rank2<MAX_LEN, DIM>, f32, D, T>) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        self.weight
            .clone()
            .put_tape(tape)
            .try_add(input)
    }
}

impl<const MAX_LEN: usize, const DIM: usize, BATCH: Dim, D: Device<f32>, T: Tape<f32, D>>
    Module<Tensor<(BATCH, Const<MAX_LEN>, Const<DIM>), f32, D, T>>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
where
    D: TensorFromVec<usize>,
    Tensor<(dfdx::shapes::Const<MAX_LEN>, dfdx::shapes::Const<DIM>), f32, D>: AsArray,
{
    type Output = Tensor<(BATCH, Const<MAX_LEN>, Const<DIM>), f32, D, T>;
    type Error = D::Err;
    fn try_forward(&self, input: Tensor<(BATCH, Const<MAX_LEN>, Const<DIM>), f32, D, T>) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        self.weight
            .clone()
            .put_tape(tape)
            .try_broadcast_like(input.shape())?
            .try_add(input)
    }
}

impl<const V: usize, const M: usize, E: Dtype, D: DeviceStorage> NonMutableModule
    for LearnedPositionalEmbedding<V, M, E, D>{}

impl<const MAX_LEN: usize, const DIM: usize, D: Device<f32>> BuildModule<D, f32>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let bound = 1. / (MAX_LEN as f32).sqrt();
        let weight = device.try_sample(Uniform::new(-bound, bound))?;
        Ok(Self {
            weight,
        })
    }
}

impl<const MAX_LEN: usize, const DIM: usize, D1: Device<f32>, D2: Device<f32>> ToDevice<D2>
    for LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D1>
{
    type Output = LearnedPositionalEmbedding<MAX_LEN, DIM, f32, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        LearnedPositionalEmbedding {
            weight: self.weight.to_device(device),
        }
    }
}
impl<const C: usize, const M: usize, D: SampleTensor<f32> + Device<f32>>
    TensorCollection<f32, D> for LearnedPositionalEmbedding<C, M, f32, D>
{
    fn iter_tensors<V: ModuleVisitor<Self, f32, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_tensor(
            "weight",
            |s| &s.weight,
            |s| &mut s.weight,
            TensorOptions::reset_with(|t| {
                let b = 1. / (C as f32).sqrt();
                t.try_fill_with_distr(Uniform::new(-b, b))
            }),
        )

    }
}