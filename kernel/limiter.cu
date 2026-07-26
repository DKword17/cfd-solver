/**
 * kernel/limiter.cu
 * TVD limiter CUDA kernels.
 *
 * Three limiters: minmod, van Leer, superbee.
 * Each: φ(r) = limiter(r) where r = Δ_upwind / Δ_local.
 */

__device__ inline float _limiter_minmod(float r) {
    return fmaxf(0.0f, fminf(1.0f, r));
}

__device__ inline float _limiter_van_leer(float r) {
    float abs_r = fabsf(r);
    return (r + abs_r) / (1.0f + abs_r + 1e-15f);
}

__device__ inline float _limiter_superbee(float r) {
    return fmaxf(0.0f, fmaxf(fminf(2.0f * r, 1.0f), fminf(r, 2.0f)));
}


extern "C" __global__ void _kernel_tvd_face_x(
    const float* __restrict__ phi,
    float* phi_f,
    int nx, int ny,
    int limiter_type
) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny) return;

    phi_f[j * (nx + 1) + 0] = phi[j * nx + 0];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nx - 1; i += gridDim.x * blockDim.x) {
        int idxL = j * nx + i, idxR = j * nx + (i + 1);
        float phi_W = (i > 0) ? phi[j * nx + (i - 1)] : phi[idxL];
        float phi_C = phi[idxL], phi_D = phi[idxR];
        float denom = phi_D - phi_C;
        float r = (phi_C - phi_W) / (denom + 1e-15f * (denom == 0.0f ? 0.0f : 1.0f));

        float limiter_val;
        switch (limiter_type) {
            case 0: limiter_val = _limiter_minmod(r); break;
            case 1: limiter_val = _limiter_van_leer(r); break;
            case 2: limiter_val = _limiter_superbee(r); break;
            default: limiter_val = 1.0f; break;
        }
        phi_f[j * (nx + 1) + (i + 1)] = phi_C + 0.5f * limiter_val * denom;
    }
    phi_f[j * (nx + 1) + nx] = phi[j * nx + (nx - 1)];
}


extern "C" __global__ void _kernel_tvd_face_y(
    const float* __restrict__ phi,
    float* phi_f,
    int nx, int ny,
    int limiter_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    phi_f[0 * nx + i] = phi[0 * nx + i];

    for (int j = blockIdx.y * blockDim.y + threadIdx.y;
         j < ny - 1; j += gridDim.y * blockDim.y) {
        int idxL = j * nx + i, idxR = (j + 1) * nx + i;
        float phi_S = (j > 0) ? phi[(j - 1) * nx + i] : phi[idxL];
        float phi_C = phi[idxL], phi_D = phi[idxR];
        float denom = phi_D - phi_C;
        float r = (phi_C - phi_S) / (denom + 1e-15f * (denom == 0.0f ? 0.0f : 1.0f));

        float limiter_val;
        switch (limiter_type) {
            case 0: limiter_val = _limiter_minmod(r); break;
            case 1: limiter_val = _limiter_van_leer(r); break;
            case 2: limiter_val = _limiter_superbee(r); break;
            default: limiter_val = 1.0f; break;
        }
        phi_f[(j + 1) * nx + i] = phi_C + 0.5f * limiter_val * denom;
    }
    phi_f[ny * nx + i] = phi[(ny - 1) * nx + i];
}


extern "C" __global__ void _kernel_limiter_value(
    const float* __restrict__ phi,
    float* r_field,
    float* phi_field,
    int nx, int ny,
    int limiter_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = j * nx + i;
    float phi_W = phi[j * nx + (i - 1)];
    float phi_C = phi[idx];
    float phi_E = phi[j * nx + (i + 1)];
    float denom = phi_E - phi_C;
    float r = (phi_C - phi_W) / (denom + 1e-15f);
    r_field[idx] = r;

    switch (limiter_type) {
        case 0: phi_field[idx] = _limiter_minmod(r); break;
        case 1: phi_field[idx] = _limiter_van_leer(r); break;
        case 2: phi_field[idx] = _limiter_superbee(r); break;
        default: phi_field[idx] = 1.0f; break;
    }
}
