use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use dfdx::{nn::modules::*, shapes::*, tensor::*, tensor_ops::*};

pub mod builder {
    #[derive(Debug, Clone)]
    pub struct MultiHeadAttention<
        const EMBED_DIM: usize,
        const NUM_HEADS: usize,
        const MAX_LEN: usize,
        const K_DIM: usize = EMBED_DIM,
        const V_DIM: usize = EMBED_DIM,
    >;
    impl<const M: usize, const H: usize, const MAX_LEN: usize, const K: usize, const V: usize>
        MultiHeadAttention<M, H, MAX_LEN, K, V>
    {
        pub const TYPE_CHECK: () = assert!(
            K % H == 0 && V % H == 0,
            "NUM_HEADS must divide K_DIM & V_DIM evenly! If you haven't specified K_DIM & V_DIM, they default to EMBED_DIM, which means NUM_HEADS must divide EMBED_DIM evenly."
        );
    }
}

impl<
        const M: usize,
        const H: usize,
        const MAX_LEN: usize,
        const K: usize,
        const V: usize,
        E: Dtype,
        D: Device<E>,
    > BuildOnDevice<D, E> for builder::MultiHeadAttention<M, H, MAX_LEN, K, V>
where
    MultiHeadAttention<M, H, MAX_LEN, K, V, E, D>: BuildModule<D, E>,
{
    type Built = MultiHeadAttention<M, H, MAX_LEN, K, V, E, D>;
    fn try_build_on_device(device: &D) -> Result<Self::Built, <D>::Err> {
        #[allow(clippy::let_unit_value)]
        let _ = Self::TYPE_CHECK;
        Self::Built::try_build(device)
    }
}

/// A multi-head attention layer.
///
/// Generics:
/// - `EMBED_DIM`: The size of query vectors.
/// - `NUM_HEADS` The number of heads to split query/key/value into.
/// - *Optional* `K_DIM`: The size of key vectors. Defaults to `EMBED_DIM`
/// - *Optional* `V_DIM` The size of value vectors. Defaults to `EMBED_DIM`
///
/// **Pytorch equivalent**: `torch.nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)`
///
/// Examples
/// - `MultiHeadAttention<8, 2>` is an attention layer with 2 heads and 8 token, key and value dims.
/// - `MultiHeadAttention<8, 2, 6, 4>` is an attention layer with the key and value dimension different
///   than the embed dimension
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<
    const EMBED_DIM: usize,
    const NUM_HEADS: usize,
    const MAX_LEN: usize,
    const K_DIM: usize,
    const V_DIM: usize,
    E: Dtype,
    D: Storage<E>,
> {
    pub w_q: UnbiasedLinear<EMBED_DIM, K_DIM, E, D>,
    pub w_k: UnbiasedLinear<EMBED_DIM, K_DIM, E, D>,
    pub w_v: UnbiasedLinear<EMBED_DIM, V_DIM, E, D>,
    pub w_o: UnbiasedLinear<V_DIM, EMBED_DIM, E, D>,
    pub biases: Tensor<Rank3<NUM_HEADS, MAX_LEN, MAX_LEN>, E, D>,
}

impl<
        const M: usize,
        const H: usize,
        const MAX_LEN: usize,
        const K: usize,
        const V: usize,
        E,
        D: Device<E>,
    > TensorCollection<E, D> for MultiHeadAttention<M, H, MAX_LEN, K, V, E, D>
where
    E: Dtype + Float + SampleUniform,
{
    type To<E2: Dtype, D2: Device<E2>> = MultiHeadAttention<M, H, MAX_LEN, K, V, E2, D2>;

    fn iter_tensors<W: ModuleVisitor<Self, E, D>>(
        visitor: &mut W,
    ) -> Result<Option<Self::To<W::E2, W::D2>>, W::Err> {
        visitor.visit_fields(
            (
                Self::module("w_q", |s| &s.w_q, |s| &mut s.w_q),
                Self::module("w_k", |s| &s.w_k, |s| &mut s.w_k),
                Self::module("w_v", |s| &s.w_v, |s| &mut s.w_v),
                Self::module("w_o", |s| &s.w_o, |s| &mut s.w_o),
                Self::tensor(
                    "biases",
                    |s| &s.biases,
                    |s| &mut s.biases,
                    TensorOptions::detached(|t| {
                        let mut mask = vec![E::zero(); H * MAX_LEN * MAX_LEN];
                        let ratio = 2_f32.powf(-(2_f32.powf(-((H as f32).log2() - 3.))));
                        for h in 0..H {
                            // ALiBi
                            let m = -ratio.powi(h as i32 + 1); // Negative to apply minus mask
                            for i in 0..MAX_LEN {
                                for j in 0..i {
                                    mask[h * MAX_LEN * MAX_LEN + i * MAX_LEN + j] =
                                        E::from_f32((i as f32 - j as f32).abs() * m).unwrap();
                                }
                            }

                            // Causal
                            for i in 0..MAX_LEN {
                                for j in i + 1..MAX_LEN {
                                    mask[h * MAX_LEN * MAX_LEN + i * MAX_LEN + j] = -E::infinity();
                                }
                            }
                        }
                        t.copy_from(&mask);

                        Ok(())
                    }),
                ),
            ),
            |(w_q, w_k, w_v, w_o, biases)| MultiHeadAttention {
                w_q,
                w_k,
                w_v,
                w_o,
                biases,
            },
        )
    }
}

impl<
        const M: usize,
        const H: usize,
        const MAX_LEN: usize,
        const K: usize,
        const V: usize,
        E,
        D,
        S1,
        S2,
        T,
    >
    Module<(
        Tensor<(S1, Const<M>), E, D, T>,
        Tensor<(S2, Const<M>), E, D>,
        Tensor<(S2, Const<M>), E, D>,
    )> for MultiHeadAttention<M, H, MAX_LEN, K, V, E, D>
where
    E: Dtype + Float,
    D: Device<E>,
    S1: Dim,
    S2: Dim,
    T: Tape<E, D>,
{
    type Output = Tensor<(S1, Const<M>), E, D, T>;
    type Error = D::Err;

    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(S1, Const<M>), E, D, T>,
            Tensor<(S2, Const<M>), E, D>,
            Tensor<(S2, Const<M>), E, D>,
        ),
    ) -> Result<Self::Output, D::Err> {
        let s1 = q.shape().0;
        let s2 = k.shape().0;
        let v = self
            .w_v
            .try_forward(v.retaped::<T>())?
            .try_reshape_like(&(s2, H, V / H))?
            .try_permute::<_, Axes3<1, 0, 2>>()?;

        let k = self
            .w_k
            .try_forward(k.retaped::<T>())?
            .try_reshape_like(&(s2, H, K / H))?
            .try_permute::<_, Axes3<1, 2, 0>>()?;

        let q = self
            .w_q
            .try_forward(q)?
            .try_reshape_like(&(s1, H, K / H))?
            .try_permute::<_, Axes3<1, 0, 2>>()?;

        // Get weights
        let scalar = E::ONE / E::from_usize(K / H).unwrap().sqrt();
        let weights = q
            .try_matmul(k)?
            .try_mul(scalar)?
            .try_add(
                self.biases
                    .clone()
                    .try_slice((.., ..s1.size(), ..s2.size()))?
                    .realize::<(usize, S1, S2)>(),
            )?
            .try_softmax::<Axis<2>>()?;

        // Get new tokens
        let tokens = weights
            .try_matmul(v)?
            .try_permute::<_, Axes3<1, 0, 2>>()?
            .try_reshape_like(&(s1, Const::<V>))?;

        self.w_o.try_forward(tokens)
    }
}

impl<
        const M: usize,
        const H: usize,
        const MAX_LEN: usize,
        const K: usize,
        const V: usize,
        E,
        D,
        B,
        S1,
        S2,
        T,
    >
    Module<(
        Tensor<(B, S1, Const<M>), E, D, T>,
        Tensor<(B, S2, Const<M>), E, D>,
        Tensor<(B, S2, Const<M>), E, D>,
    )> for MultiHeadAttention<M, H, MAX_LEN, K, V, E, D>
where
    E: Dtype + Float,
    D: Device<E>,
    B: Dim,
    S1: Dim,
    S2: Dim,
    T: Tape<E, D>,
{
    type Output = Tensor<(B, S1, Const<M>), E, D, T>;
    type Error = D::Err;

    /// Batched Encoder-Decoder style self attention where one set of tensors is used for values and keys, and another is used for queries
    fn try_forward(
        &self,
        (q, k, v): (
            Tensor<(B, S1, Const<M>), E, D, T>,
            Tensor<(B, S2, Const<M>), E, D>,
            Tensor<(B, S2, Const<M>), E, D>,
        ),
    ) -> Result<Self::Output, D::Err> {
        let b = q.shape().0;
        let s1 = q.shape().1;
        let s2 = v.shape().1;

        let v = self
            .w_v
            .try_forward(v.retaped::<T>())?
            .try_reshape_like(&(b, s2, H, V / H))?
            .try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        let k = self
            .w_k
            .try_forward(k.retaped::<T>())?
            .try_reshape_like(&(b, s2, H, K / H))?
            .try_permute::<_, Axes4<0, 2, 3, 1>>()?;

        let q = self
            .w_q
            .try_forward(q)?
            .try_reshape_like(&(b, s1, H, K / H))?
            .try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        // Get weights
        let scalar = E::ONE / E::from_usize(K / H).unwrap().sqrt();
        let weights = q
            .try_matmul(k)?
            .try_mul(scalar)?
            .try_add(
                self.biases
                    .clone()
                    .try_slice((.., ..s1.size(), ..s2.size()))?
                    .realize::<(usize, S1, S2)>()
                    .try_broadcast_like(&(b, H, s1, s2))?,
            )?
            .try_softmax::<Axis<3>>()?;

        // Get new tokens
        let tokens = weights
            .try_matmul(v)?
            .try_permute::<_, Axes4<0, 2, 1, 3>>()?
            .try_reshape_like(&(b, s1, Const::<V>))?;

        self.w_o.try_forward(tokens)
    }
}

impl<
        const M: usize,
        const H: usize,
        const MAX_LEN: usize,
        const K: usize,
        const V: usize,
        E,
        D,
        Src,
    > Module<Src> for MultiHeadAttention<M, H, MAX_LEN, K, V, E, D>
where
    E: Dtype,
    D: Device<E>,
    Src: SplitTape,
    Self: Module<(Src, Src::NoTape, Src::NoTape), Output = Src, Error = D::Err>,
{
    type Output = Src;
    type Error = D::Err;

    fn try_forward(&self, src: Src) -> Result<Self::Output, D::Err> {
        let (src, tape) = src.split_tape();
        self.try_forward((src.clone().put_tape(tape), src.clone(), src))
    }
}

impl<
        const M: usize,
        const H: usize,
        const MAX_LEN: usize,
        const K: usize,
        const V: usize,
        E: Dtype,
        D: Device<E>,
    > NonMutableModule for MultiHeadAttention<M, H, MAX_LEN, K, V, E, D>
{
}
