/**
 * kernel/laplace_cuda.cu
 * Poisson equation CUDA solver — Red-Black Gauss-Seidel
 *
 * Solves: a_P·p'_P = Σ(a_NB·p'_NB) - b
 *
 * Red-black coloring removes data dependency for GPU parallelism.
 * tile=16×16 (256 threads/block) for optimal occupancy.
 */

extern "C" __global__ void _kernel_poisson_rbgs(
    float* __restrict__ p,
    const float* __restrict__ rhs,
    const float* __restrict__ aE,
    const float* __restrict__ aW,
    const float* __restrict__ aN,
    const float* __restrict__ aS,
    const float* __restrict__ aP,
    int nx, int ny,
    int red  // 0=black, 1=red
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1) return;
    int is_red = ((i + j) & 1) == 0;
    if (is_red != red) return;

    int idx = j * nx + i;
    float p_new = (aE[idx] * p[idx + 1] + aW[idx] * p[idx - 1] +
                   aN[idx] * p[idx + nx] + aS[idx] * p[idx - nx] -
                   rhs[idx]) / aP[idx];
    p[idx] = p_new;
}


extern "C" __global__ void _kernel_poisson_residual(
    const float* __restrict__ p,
    const float* __restrict__ rhs,
    const float* __restrict__ aE,
    const float* __restrict__ aW,
    const float* __restrict__ aN,
    const float* __restrict__ aS,
    const float* __restrict__ aP,
    int nx, int ny,
    float* residual_norm
) {
    extern __shared__ float s_res[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int i = gid % nx;
    int j = gid / nx;

    float local_res = 0.0f;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = j * nx + i;
        float Ax = aP[idx] * p[idx]
                 - aE[idx] * p[idx + 1]
                 - aW[idx] * p[idx - 1]
                 - aN[idx] * p[idx + nx]
                 - aS[idx] * p[idx - nx];
        local_res = rhs[idx] - Ax;
        local_res = local_res * local_res;
    }

    s_res[tid] = local_res;
    __syncthreads();

    // shared memory tree reduction down to 32 elements
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) s_res[tid] += s_res[tid + s];
        __syncthreads();
    }

    // warp shuffle for final 32
    if (tid < 32) {
        float val = s_res[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (tid == 0) atomicAdd(residual_norm, val);
    }
}


extern "C" __global__ void _kernel_poisson_coefficients(
    float* aE, float* aW, float* aN, float* aS, float* aP,
    int nx, int ny,
    float dx, float dy,
    float dt, float rho
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;
    float coeff = dt / rho;
    float ae_val = coeff * dy / dx;
    float an_val = coeff * dx / dy;

    aE[idx] = ae_val;
    aW[idx] = ae_val;
    aN[idx] = an_val;
    aS[idx] = an_val;
    aP[idx] = ae_val * 2.0f + an_val * 2.0f;
}
