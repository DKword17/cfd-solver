/**
 * kernel/face_interp.cu
 * Rhie-Chow interpolation CUDA kernels.
 *
 * Collocated grid face interpolation with pressure gradient correction
 * to prevent checkerboard pressure oscillations.
 */

extern "C" __global__ void _kernel_rhie_chow_faces_x(
    const float* __restrict__ phi_c,
    const float* __restrict__ grad_x,
    const float* __restrict__ Df,
    float* phi_f,
    int nx, int ny,
    float dx
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (f > nx || j >= ny) return;

    if (f == 0) {
        phi_f[j * (nx + 1) + f] = phi_c[j * nx + 0];
    } else if (f == nx) {
        phi_f[j * (nx + 1) + f] = phi_c[j * nx + (nx - 1)];
    } else {
        int iL = f - 1, iR = f;
        int idxL = j * nx + iL, idxR = j * nx + iR;
        float phi_avg = 0.5f * (phi_c[idxL] + phi_c[idxR]);
        float D_avg   = 0.5f * (Df[idxL] + Df[idxR]);
        float grad_avg_x = 0.5f * (grad_x[idxL] + grad_x[idxR]);
        float p_grad_term = (phi_c[idxR] - phi_c[idxL]) - grad_avg_x * dx;
        phi_f[j * (nx + 1) + f] = phi_avg - D_avg * p_grad_term;
    }
}


extern "C" __global__ void _kernel_rhie_chow_faces_y(
    const float* __restrict__ phi_c,
    const float* __restrict__ grad_y,
    const float* __restrict__ Df,
    float* phi_f,
    int nx, int ny,
    float dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || f > ny) return;

    if (f == 0) {
        phi_f[f * nx + i] = phi_c[0 * nx + i];
    } else if (f == ny) {
        phi_f[f * nx + i] = phi_c[(ny - 1) * nx + i];
    } else {
        int jL = f - 1, jR = f;
        int idxL = jL * nx + i, idxR = jR * nx + i;
        float phi_avg = 0.5f * (phi_c[idxL] + phi_c[idxR]);
        float D_avg   = 0.5f * (Df[idxL] + Df[idxR]);
        float grad_avg_y = 0.5f * (grad_y[idxL] + grad_y[idxR]);
        float p_grad_term = (phi_c[idxR] - phi_c[idxL]) - grad_avg_y * dy;
        phi_f[f * nx + i] = phi_avg - D_avg * p_grad_term;
    }
}


extern "C" __global__ void _kernel_compute_Df(
    const float* __restrict__ aP,
    float* Df,
    int n_cells,
    float cell_volume
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells) return;
    Df[idx] = cell_volume / aP[idx];
}
