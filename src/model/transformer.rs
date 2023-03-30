use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use dfdx::{
    nn::modules::*,
    shapes::Dtype,
    tensor::{DeviceStorage, PutTape, SplitTape, ToDevice},
    tensor_ops::{Device, TryAdd},
};

use super::mha::MultiHeadAttention;

pub type TransformerEncoder<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    const NUM_LAYERS: usize,
    const MAX_LEN: usize,
    E,
    D,
> = Repeated<TransformerEncoderBlock<MODEL_DIM, NUM_HEADS, FF_DIM, MAX_LEN, E, D>, NUM_LAYERS>;

pub mod builder {
    #[derive(Debug)]
    pub struct TransformerEncoder<
        const MODEL_DIM: usize,
        const NUM_HEADS: usize,
        const FF_DIM: usize,
        const NUM_LAYERS: usize,
        const MAX_LEN: usize,
    >;

    #[derive(Debug)]
    pub struct TransformerEncoderBlock<
        const MODEL_DIM: usize,
        const NUM_HEADS: usize,
        const FF_DIM: usize,
        const MAX_LEN: usize,
    >;
}

impl<
        const M: usize,
        const H: usize,
        const F: usize,
        const L: usize,
        const MAX_LEN: usize,
        E: Dtype,
        D: Device<E>,
    > BuildOnDevice<D, E> for builder::TransformerEncoder<M, H, F, L, MAX_LEN>
where
    TransformerEncoder<M, H, F, L, MAX_LEN, E, D>: BuildModule<D, E>,
{
    type Built = TransformerEncoder<M, H, F, L, MAX_LEN, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

impl<
        const M: usize,
        const H: usize,
        const F: usize,
        const MAX_LEN: usize,
        E: Dtype,
        D: Device<E>,
    > BuildOnDevice<D, E> for builder::TransformerEncoderBlock<M, H, F, MAX_LEN>
where
    TransformerEncoderBlock<M, H, F, MAX_LEN, E, D>: BuildModule<D, E>,
{
    type Built = TransformerEncoderBlock<M, H, F, MAX_LEN, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        Self::Built::try_build(device)
    }
}

#[derive(Clone, Debug)]
pub struct TransformerEncoderBlock<
    const MODEL_DIM: usize,
    const NUM_HEADS: usize,
    const FF_DIM: usize,
    const MAX_LEN: usize,
    E: Dtype,
    D: DeviceStorage,
> {
    pub self_attn: MultiHeadAttention<MODEL_DIM, NUM_HEADS, MAX_LEN, MODEL_DIM, MODEL_DIM, E, D>,
    pub norm1: LayerNorm1D<MODEL_DIM, E, D>,
    pub ff: FF<MODEL_DIM, FF_DIM, E, D>,
    pub norm2: LayerNorm1D<MODEL_DIM, E, D>,
}

type FF<const M: usize, const F: usize, E, D> =
    Residual<(UnbiasedLinear<M, F, E, D>, GeLU, UnbiasedLinear<F, M, E, D>)>;

impl<const M: usize, const H: usize, const F: usize, const MAX_LEN: usize, E, D: Device<E>>
    BuildModule<D, E> for TransformerEncoderBlock<M, H, F, MAX_LEN, E, D>
where
    E: Dtype + Float + SampleUniform,
{
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self {
            self_attn: BuildModule::try_build(device)?,
            norm1: BuildModule::try_build(device)?,
            ff: BuildModule::try_build(device)?,
            norm2: BuildModule::try_build(device)?,
        })
    }
}

impl<const M: usize, const H: usize, const F: usize, const MAX_LEN: usize, E, D: Device<E>>
    TensorCollection<E, D> for TransformerEncoderBlock<M, H, F, MAX_LEN, E, D>
where
    E: Dtype + Float + SampleUniform,
{
    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(visitor: &mut V) -> Result<(), V::Err> {
        visitor.visit_module("self_attn", |s| &s.self_attn, |s| &mut s.self_attn)?;
        visitor.visit_module("norm1", |s| &s.norm1, |s| &mut s.norm1)?;
        visitor.visit_module("ff", |s| &s.ff, |s| &mut s.ff)?;
        visitor.visit_module("norm2", |s| &s.norm2, |s| &mut s.norm2)
    }
}

impl<
        const M: usize,
        const H: usize,
        const F: usize,
        const MAX_LEN: usize,
        E: Dtype,
        D1: Device<E>,
        D2: Device<E>,
    > ToDevice<D2> for TransformerEncoderBlock<M, H, F, MAX_LEN, E, D1>
{
    type Output = TransformerEncoderBlock<M, H, F, MAX_LEN, E, D2>;
    fn to_device(&self, device: &D2) -> Self::Output {
        TransformerEncoderBlock {
            self_attn: self.self_attn.to_device(device),
            norm1: self.norm1.to_device(device),
            ff: self.ff.to_device(device),
            norm2: self.norm2.to_device(device),
        }
    }
}

impl<
        const M: usize,
        const H: usize,
        const F: usize,
        const MAX_LEN: usize,
        E: Dtype,
        D: Device<E>,
        Src,
    > Module<Src> for TransformerEncoderBlock<M, H, F, MAX_LEN, E, D>
where
    Src: SplitTape + TryAdd<Src::NoTape, Err = D::Err>,
    MultiHeadAttention<M, H, MAX_LEN, M, M, E, D>: Module<Src, Output = Src, Error = D::Err>,
    LayerNorm1D<M, E, D>: Module<Src, Output = Src, Error = D::Err>,
    FF<M, F, E, D>: Module<Src, Output = Src, Error = D::Err>,
{
    type Output = Src;
    type Error = D::Err;

    fn try_forward(&self, src: Src) -> Result<Self::Output, D::Err> {
        let (src, tape) = src.split_tape();
        let x = self
            .self_attn
            .try_forward(self.norm1.try_forward(src.clone().put_tape(tape))?)?
            .try_add(src)?;
        let (x, tape) = x.split_tape();
        self.ff
            .try_forward(self.norm2.try_forward(x.clone().put_tape(tape))?)?
            .try_add(x)
    }
}

impl<
        const M: usize,
        const H: usize,
        const F: usize,
        const MAX_LEN: usize,
        E: Dtype,
        D: Device<E>,
    > NonMutableModule for TransformerEncoderBlock<M, H, F, MAX_LEN, E, D>
{
}
