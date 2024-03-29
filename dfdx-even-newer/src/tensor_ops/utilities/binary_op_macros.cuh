#include "cuda_utils.cuh"

#define LONG_BINARY_OP(TYPENAME, FORWARD, BACKWARD_LHS, BACKWARD_RHS, OP_STRUCT, FUNC, DFDX, DFDY) \
extern "C" __global__ void FORWARD( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const TYPENAME *lhs, \
    const size_t *lhs_strides, \
    const TYPENAME *rhs, \
    const size_t *rhs_strides, \
    TYPENAME *out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
\
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides); \
\
    TYPENAME x = lhs[lhs_i]; \
    TYPENAME y = rhs[rhs_i]; \
    TYPENAME fx; \
\
    FUNC\
\
    out[i] = fx; \
} \
\
extern "C" __global__ void BACKWARD_LHS( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *out_strides, \
    const TYPENAME *lhs, \
    TYPENAME *grad_lhs, \
    const size_t chunk_len, \
    const TYPENAME *rhs, \
    const size_t *rhs_strides, \
    const TYPENAME *grad_out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
\
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
\
    unsigned int lhs_i = i / chunk_len; \
    unsigned int rhs_i = get_strided_index(i, num_dims, dims, rhs_strides); \
\
    TYPENAME x = lhs[lhs_i]; \
    TYPENAME y = rhs[rhs_i]; \
    TYPENAME go = grad_out[out_i]; \
\
    TYPENAME dfdx = (DFDX); \
\
    chunk_sum(chunk_len, dfdx * go, grad_lhs); \
} \
\
extern "C" __global__ void BACKWARD_RHS( \
    const OP_STRUCT op, \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims, \
    const size_t *out_strides, \
    const TYPENAME *lhs, \
    const size_t *lhs_strides, \
    const TYPENAME *rhs, \
    TYPENAME *grad_rhs, \
    const size_t chunk_len, \
    const TYPENAME *grad_out \
) { \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= numel) { \
        return; \
    } \
\
    unsigned int out_i = get_strided_index(i, num_dims, dims, out_strides); \
\
    unsigned int lhs_i = get_strided_index(i, num_dims, dims, lhs_strides); \
    unsigned int rhs_i = i / chunk_len; \
\
    TYPENAME x = lhs[lhs_i]; \
    TYPENAME y = rhs[rhs_i]; \
    TYPENAME go = grad_out[out_i]; \
\
    TYPENAME dfdy = (DFDY); \
\
    chunk_sum(chunk_len, dfdy * go, grad_rhs); \
}

#define BINARY_OP(TYPENAME, FORWARD, BACKWARD_LHS, BACKWARD_RHS, OP_STRUCT, FUNC, DFDX, DFDY) \
    LONG_BINARY_OP(TYPENAME, FORWARD, BACKWARD_LHS, BACKWARD_RHS, OP_STRUCT, fx = (FUNC);, DFDX, DFDY)
