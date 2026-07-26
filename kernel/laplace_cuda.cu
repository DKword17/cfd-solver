/**
 * kernel/laplace_cuda.cu
 * 压力 Poisson 方程 CUDA 求解器
 *
 * 求解: ∇·(Δt/ρ · ∇p') = ∇·u*   (SIMPLE 算法 Step 2)
 *       即  α·(∂²p'/∂x² + ∂²p'/∂y²) = b
 *
 * 离散: 5 点差分, 均匀网格
 *       a_P·p'_P = a_E·p'_E + a_W·p'_W + a_N·p'_N + a_S·p'_S - b
 *
 * 求解器: 红黑 Gauss-Seidel (RBSOR)
 *   红黑着色消除数据依赖, 适合 GPU warp 执行.
 *   tile=16×16 (256 threads/block), occupancy 最优.
 */

/* ═══════════════════════════════════════════════════════════════════
   核 1: 红黑 Gauss-Seidel 一次迭代
   直接用全局内存, 不做共享内存 tiling.
   为什么? 简化调试, 5 点 stencil 的 L1/L2 缓存命中率已足够高.
   后续优化方向: 加共享内存 halo 减少全局访存 ~30%.

   参数:
     p:       压力修正 p' (输入/输出), 行主序 [ny][nx]
     rhs:     右端项 b
     aE,aW,aN,aS,aP: 5-对角系数
     nx,ny:   网格维数
     red:     0=黑点, 1=红点
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_poisson_rbgs(
    float* __restrict__ p,
    const float* __restrict__ rhs,
    const float* __restrict__ aE,
    const float* __restrict__ aW,
    const float* __restrict__ aN,
    const float* __restrict__ aS,
    const float* __restrict__ aP,
    int nx, int ny,
    int red  // 0=黑点, 1=红点
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 跳过边界 (i=0,i=nx-1,j=0,j=ny-1) 和不需要的颜色
    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1) return;

    // 红: (i+j) 偶; 黑: (i+j) 奇
    int is_red = ((i + j) & 1) == 0;
    if (is_red != red) return;

    int idx = j * nx + i;

    // Gauss-Seidel 更新: p_new = (aE·pE + aW·pW + aN·pN + aS·pS - b) / aP
    float p_new = (aE[idx] * p[idx + 1] + aW[idx] * p[idx - 1] +
                   aN[idx] * p[idx + nx] + aS[idx] * p[idx - nx] -
                   rhs[idx]) / aP[idx];

    p[idx] = p_new;
}


/* ═══════════════════════════════════════════════════════════════════
   核 2: L2 残差规约 (warp-level reduce)
   每个 block 处理一块, 树形归约 -> atomicAdd 跨 block.
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_poisson_residual(
    const float* __restrict__ p,
    const float* __restrict__ rhs,
    const float* __restrict__ aE,
    const float* __restrict__ aW,
    const float* __restrict__ aN,
    const float* __restrict__ aS,
    const float* __restrict__ aP,
    int nx, int ny,
    float* residual_norm  // 输出: L2 残差
) {
    // 1D block: 每个线程处理一个 cell
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int i = gid % nx;
    int j = gid / nx;

    float local_res = 0.0f;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        int idx = j * nx + i;
        // 残差: r = b - A·p
        float Ax = aP[idx] * p[idx]
                 - aE[idx] * p[idx + 1]
                 - aW[idx] * p[idx - 1]
                 - aN[idx] * p[idx + nx]
                 - aS[idx] * p[idx - nx];
        local_res = rhs[idx] - Ax;
        local_res = local_res * local_res;
    }

    // warp-level 树形归约
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        local_res += __shfl_xor_sync(0xFFFFFFFF, local_res, offset);
    }

    if (tid == 0) {
        atomicAdd(residual_norm, local_res);
    }
}


/* ═══════════════════════════════════════════════════════════════════
   核 3: 系数矩阵组装
   为什么系数不变? 因为 α=Δt/ρ 在 SIMPLE 迭代内是常数.
   如果 dt 不变, 只需在初始时调用一次.
   ─────────────────────────────────────────────────────────────────── */
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
