use dfdx::prelude::tensor_collection::{ModuleVisitor, TensorCollection, TensorOptions};
use dfdx::prelude::*;
use num::Float;
use rand_distr::uniform::SampleUniform;
use rand_distr::Uniform;

pub mod builder {
    use std::marker::PhantomData;

    #[derive(Debug)]
    pub struct ReverseEmbedding<const VOCAB: usize, const DIM: usize, I>(PhantomData<I>);
}

impl<const V: usize, const M: usize, E: Dtype, D: Device<E>, I: BuildOnDevice<D, E>>
    BuildOnDevice<D, E> for builder::ReverseEmbedding<V, M, I>
where
    ReverseEmbedding<V, M, E, D, <I as BuildOnDevice<D, E>>::Built>: BuildModule<D, E>,
{
    type Built = ReverseEmbedding<V, M, E, D, <I as BuildOnDevice<D, E>>::Built>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, D::Err> {
        Self::Built::try_build(device)
    }
}

#[derive(Debug, Clone)]
pub struct ReverseEmbedding<const VOCAB: usize, const DIM: usize, E: Dtype, D: DeviceStorage, I> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<VOCAB, DIM>, E, D>,
    pub inner: I,
}

impl<const V: usize, const M: usize, E: Dtype, D: DeviceStorage, I> NonMutableModule
    for ReverseEmbedding<V, M, E, D, I>
{
}

impl<
        const V: usize,
        const M: usize,
        E: Dtype + Float + SampleUniform,
        D: Device<E>,
        I: BuildModule<D, E>,
    > BuildModule<D, E> for ReverseEmbedding<V, M, E, D, I>
{
    fn try_build(device: &D) -> Result<Self, D::Err> {
        let bound = E::ONE / E::from_usize(V).unwrap().sqrt();
        let weight = device.try_sample(Uniform::new(-bound, bound))?;
        Ok(Self {
            weight,
            inner: I::try_build(device)?,
        })
    }
}

impl<
        const C: usize,
        const M: usize,
        E: Dtype + Float + SampleUniform,
        D: SampleTensor<E>,
        I: TensorCollection<E, D>,
    > TensorCollection<E, D> for ReverseEmbedding<C, M, E, D, I>
{
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_module("inner", |s| &s.inner, |s| &mut s.inner)?;
        visitor.visit_tensor(
            "weight",
            |s| &s.weight,
            |s| &mut s.weight,
            TensorOptions::reset_with(|t| {
                let b: E = E::ONE / E::from_usize(C).unwrap().sqrt();
                t.try_fill_with_distr(Uniform::new(-b, b))
            }),
        )
    }
}

impl<
        const V: usize,
        const M: usize,
        SEQ: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
        I: Module<
            Tensor<(SEQ, Const<M>), E, D, T>,
            Output = Tensor<(SEQ, Const<M>), E, D, T>,
            Error = D::Err,
        >,
    > Module<Tensor<(SEQ,), usize, D, T>> for ReverseEmbedding<V, M, E, D, I>
{
    type Output = Tensor<(SEQ, Const<V>), E, D, T>;
    type Error = D::Err;

    fn try_forward(&self, input: Tensor<(SEQ,), usize, D, T>) -> Result<Self::Output, D::Err> {
        let (input, tape) = input.split_tape();
        let embedded = self.weight.clone().put_tape(tape).try_gather(input)?;
        let output = self.inner.try_forward(embedded)?;
        output.try_matmul(self.weight.retaped::<T>().permute())
    }
}

impl<
        const VOCAB: usize,
        const DIM: usize,
        BATCH: Dim,
        SEQ: Dim,
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
        I: Module<
            Tensor<(BATCH, SEQ, Const<DIM>), E, D, T>,
            Output = Tensor<(BATCH, SEQ, Const<DIM>), E, D, T>,
            Error = D::Err,
        >,
    > Module<Tensor<(BATCH, SEQ), usize, D, T>> for ReverseEmbedding<VOCAB, DIM, E, D, I>
{
    type Output = Tensor<(BATCH, SEQ, Const<VOCAB>), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(BATCH, SEQ), usize, D, T>,
    ) -> Result<Self::Output, D::Err> {
        let (input, tape) = input.split_tape();
        let embedded = self.weight.clone().put_tape(tape).try_gather(input)?;
        let output = self.inner.try_forward(embedded)?;
        output.try_matmul(self.weight.retaped::<T>().permute())
    }
}

impl<
        const VOCAB: usize,
        const DIM: usize,
        E: Dtype,
        D1: Device<E>,
        D2: Device<E>,
        I: ToDevice<D2>,
    > ToDevice<D2> for ReverseEmbedding<VOCAB, DIM, E, D1, I>
{
    type Output = ReverseEmbedding<VOCAB, DIM, E, D2, <I as ToDevice<D2>>::Output>;
    fn to_device(&self, device: &D2) -> Self::Output {
        ReverseEmbedding {
            weight: self.weight.to_device(device),
            inner: self.inner.to_device(device),
        }
    }
}
