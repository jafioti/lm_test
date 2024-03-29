use crate::tensor_ops::cpu_kernels::UnaryDerivative;

impl<F: num_traits::Float> UnaryDerivative<F> for super::NegateKernelOp {
    #[inline(always)]
    fn f(&self, x: &F) -> F {
        x.neg()
    }
    #[inline(always)]
    fn df(&self, _: &F) -> F {
        F::one().neg()
    }
}
