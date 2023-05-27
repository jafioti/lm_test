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
pub struct ReverseEmbedding<const VOCAB: usize, const DIM: usize, E: Dtype, D: Storage<E>, I> {
    /// Transposed weight matrix, shape (I, O)
    pub weight: Tensor<Rank2<VOCAB, DIM>, E, D>,
    pub inner: I,
}

impl<const V: usize, const M: usize, E: Dtype, D: Storage<E>, I> NonMutableModule
    for ReverseEmbedding<V, M, E, D, I>
{
}

impl<
        const C: usize,
        const M: usize,
        E: Dtype + Float + SampleUniform,
        D: Device<E>,
        I: TensorCollection<E, D>,
    > TensorCollection<E, D> for ReverseEmbedding<C, M, E, D, I>
{
    type To<E2: Dtype, D2: Device<E2>> = ReverseEmbedding<C, M, E2, D2, I::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                Self::module("inner", |s| &s.inner, |s| &mut s.inner),
                Self::tensor(
                    "weight",
                    |s| &s.weight,
                    |s| &mut s.weight,
                    TensorOptions::reset_with(|t| {
                        let b: E = E::ONE / E::from_usize(C).unwrap().sqrt();
                        t.try_fill_with_distr(Uniform::new(-b, b))
                    }),
                ),
            ),
            |(inner, weight)| ReverseEmbedding { inner, weight },
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
        output.try_matmul(self.weight.clone().try_permute()?)
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
        output.try_matmul(self.weight.clone().try_permute()?)
    }
}
