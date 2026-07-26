/**
 * kernel/gradient.cu
 * Gradient computation GPU kernels.
 *
 * Green-Gauss cell-centered gradient, central difference,
 * divergence, and vorticity.
 */

extern "C" __global__ void _kernel_gradient_gg(
    const float* __restrict__ phi_f_x,
    const float* __restrict__ phi_f_y,
    float* grad_x,
    float* grad_y,
    int nx, int ny,
    float inv_dx, float inv_dy,
    float inv_volume
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;
    float phi_e = phi_f_x[j * (nx + 1) + (i + 1)];
    float phi_w = phi_f_x[j * (nx + 1) + i];
    float phi_n = phi_f_y[(j + 1) * nx + i];
    float phi_s = phi_f_y[j * nx + i];

    grad_x[idx] = (phi_e - phi_w) * inv_dx;
    grad_y[idx] = (phi_n - phi_s) * inv_dy;
}


extern "C" __global__ void _kernel_gradient_central(
    const float* __restrict__ phi,
    float* grad_x,
    float* grad_y,
    int nx, int ny,
    float inv_dx, float inv_dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = j * nx + i;
    grad_x[idx] = (phi[idx + 1] - phi[idx - 1]) * 0.5f * inv_dx;
    grad_y[idx] = (phi[(j + 1) * nx + i] - phi[(j - 1) * nx + i]) * 0.5f * inv_dy;
}


extern "C" __global__ void _kernel_divergence(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* div,
    int nx, int ny,
    float inv_dx, float inv_dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = j * nx + i;
    float du_dx = (u[idx + 1] - u[idx - 1]) * 0.5f * inv_dx;
    float dv_dy = (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) * 0.5f * inv_dy;
    div[idx] = du_dx + dv_dy;
}


extern "C" __global__ void _kernel_vorticity(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* omega,
    int nx, int ny,
    float inv_dx, float inv_dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = j * nx + i;
    float dv_dx = (v[idx + 1] - v[idx - 1]) * 0.5f * inv_dx;
    float du_dy = (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) * 0.5f * inv_dy;
    omega[idx] = dv_dx - du_dy;
}
