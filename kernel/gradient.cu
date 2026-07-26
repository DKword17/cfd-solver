/**
 * kernel/gradient.cu
 * 梯度计算 GPU 核
 *
 * 实现:
 *   1. Green-Gauss 体心梯度
 *      ∇φ_C = (1/V) · Σ_f φ_f · n_f · A_f
 *      其中 V 是控制体体积, φ_f 是面值, n_f 是单位法向量
 *
 *   2. 速度散度 ∇·u (连续方程残差)
 *   3. 涡量 ω_z = ∂v/∂x - ∂u/∂y
 *
 * 为什么用 Green-Gauss 而不是 Least-Squares?
 *   - GG 实现简单, 均匀网格上精度与 LS 相同 (2 阶)
 *   - GG 每个面只需要一次插值, 计算量小约 30%
 *   - LS 在歪斜网格上更鲁棒, 但本项目目前是均匀网格
 *
 * tile=256: 每个线程处理一个网格点, 16×16 2D block.
 * 不需要共享内存—每个点的梯度计算只读自身和面值.
 */

/* ═══════════════════════════════════════════════════════════════════
   核 1: Green-Gauss 体心梯度 (2D)
   输入:
     phi_f_x:  x 方向面值 [ny][nx+1]
     phi_f_y:  y 方向面值 [ny+1][nx]
   输出:
     grad_x:   ∂φ/∂x [ny][nx]
     grad_y:   ∂φ/∂y [ny][nx]
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_gradient_gg(
    const float* __restrict__ phi_f_x,  // x 方向面值
    const float* __restrict__ phi_f_y,  // y 方向面值
    float* grad_x,                      // 输出: x 梯度
    float* grad_y,                      // 输出: y 梯度
    int nx, int ny,
    float inv_dx, float inv_dy,         // 1/Δx, 1/Δy
    float inv_volume                    // 1/cell_volume
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;

    // Green-Gauss: ∇φ = (1/V) · Σ_f φ_f · A_f · n_f
    // 对均匀网格, 简化为:
    //   ∂φ/∂x ≈ (φ_e - φ_w) · Δy / V = (φ_e - φ_w) / Δx
    //   ∂φ/∂y ≈ (φ_n - φ_s) · Δx / V = (φ_n - φ_s) / Δy

    // 东面 φ_e = phi_f_x[j*(nx+1) + (i+1)]
    // 西面 φ_w = phi_f_x[j*(nx+1) + i]
    // 北面 φ_n = phi_f_y[(j+1)*nx + i]
    // 南面 φ_s = phi_f_y[j*nx + i]

    float phi_e = phi_f_x[j * (nx + 1) + (i + 1)];
    float phi_w = phi_f_x[j * (nx + 1) + i];
    float phi_n = phi_f_y[(j + 1) * nx + i];
    float phi_s = phi_f_y[j * nx + i];

    // Green-Gauss 积分
    grad_x[idx] = (phi_e - phi_w) * inv_dx;   // = (phi_e - phi_w) / Δx
    grad_y[idx] = (phi_n - phi_s) * inv_dy;   // = (phi_n - phi_s) / Δy
}


/* ═══════════════════════════════════════════════════════════════════
   核 2: 直接从体心值算梯度 (2 阶中心差分)
   当不需要面值时可免去插值开销.
   — 适用于 debug 或低精度要求.
   ─────────────────────────────────────────────────────────────────── */
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

    // 中心差分: O(Δx²)
    grad_x[idx] = (phi[idx + 1] - phi[idx - 1]) * 0.5f * inv_dx;
    grad_y[idx] = (phi[(j + 1) * nx + i] - phi[(j - 1) * nx + i]) * 0.5f * inv_dy;
}


/* ═══════════════════════════════════════════════════════════════════
   核 3: 连续方程散度 ∇·u = ∂u/∂x + ∂v/∂y
   用于判断 SIMPLE 收敛和残差输出.
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_divergence(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* div,         // 输出: 散度 [ny][nx]
    int nx, int ny,
    float inv_dx, float inv_dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = j * nx + i;

    // ∂u/∂x ≈ (u_{i+1,j} - u_{i-1,j}) / (2Δx)
    // ∂v/∂y ≈ (v_{i,j+1} - v_{i,j-1}) / (2Δy)
    float du_dx = (u[idx + 1] - u[idx - 1]) * 0.5f * inv_dx;
    float dv_dy = (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) * 0.5f * inv_dy;

    div[idx] = du_dx + dv_dy;
}


/* ═══════════════════════════════════════════════════════════════════
   核 4: 涡量 ω_z = ∂v/∂x - ∂u/∂y
   用于流场可视化 (涡结构识别) 和 Q 准则.
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_vorticity(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* omega,       // 输出: 涡量 [ny][nx]
    int nx, int ny,
    float inv_dx, float inv_dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = j * nx + i;

    // ∂v/∂x ≈ (v_{i+1,j} - v_{i-1,j}) / (2Δx)
    // ∂u/∂y ≈ (u_{i,j+1} - u_{i,j-1}) / (2Δy)
    float dv_dx = (v[idx + 1] - v[idx - 1]) * 0.5f * inv_dx;
    float du_dy = (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) * 0.5f * inv_dy;

    omega[idx] = dv_dx - du_dy;
}


/* ═══════════════════════════════════════════════════════════════════
   核 5: 面值梯度插值 (体心梯度 → 面梯度)
   用于 Rhie-Chow 插值中的 ∇p̄_f 项.
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_grad_interp_faces_x(
    const float* __restrict__ grad,  // 体心梯度 [ny][nx]
    float* grad_f,                    // 面梯度 [ny][nx+1]
    int nx, int ny
) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny) return;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i <= nx;
         i += gridDim.x * blockDim.x) {
        if (i == 0) {
            grad_f[j * (nx + 1) + i] = grad[j * nx + 0];
        } else if (i == nx) {
            grad_f[j * (nx + 1) + i] = grad[j * nx + (nx - 1)];
        } else {
            grad_f[j * (nx + 1) + i] = 0.5f * (grad[j * nx + (i - 1)]
                                                + grad[j * nx + i]);
        }
    }
}
