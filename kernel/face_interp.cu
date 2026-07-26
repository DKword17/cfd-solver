/**
 * kernel/face_interp.cu
 * Rhie-Chow 插值 CUDA 核
 *
 * 同位网格 (collocated grid) 上,
 * 直接线性插值面速度会导致压力棋盘振荡 (checkerboard).
 *
 * Rhie & Chow (1983) 的思路:
 *   面速度 = 线性插值邻域速度 + 压力梯度修正
 *
 *   u_f = 0.5·(u_C + u_N) - D_f·(p_N - p_C - 0.5·(∇p_C+∇p_N)·d)
 *   其中 D_f = 体积/系数 是动量方程的通量系数
 *
 * 为什么选 256 线程/block?
 *   面插值每个线程只算一个面, 读写模式规整, 不需要共享内存,
 *   所以用 256 线程 (16×16 或 32×8) 最大化 occupancy 即可.
 *   这里用 1D block (256×1) + 1D grid, 每个线程处理一个面.
 */

/* ═══════════════════════════════════════════════════════════════════
   核 1: Rhie-Chow 东/西面插值
   输入:
     phi_c:   体心值 [ny][nx]
     grad_x:  φ 的 x 梯度 [ny][nx]
     Df:      通量系数 (体积/aP) [ny][nx]
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_rhie_chow_faces_x(
    const float* __restrict__ phi_c,   // 体心值
    const float* __restrict__ grad_x,  // 体心 x 梯度
    const float* __restrict__ Df,      // 通量系数 = 体积 / aP
    float* phi_f,                      // 输出: 面值 [ny][nx+1] (东/西面)
    int nx, int ny,
    float dx
) {
    // 面索引: f 对应 (i+1/2, j), 范围 [0, nx] × [0, ny-1]
    // 使用 1D block, 每个线程算一个面
    int f = blockIdx.x * blockDim.x + threadIdx.x;  // 面索引沿 x
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // j 方向
    if (f > nx || j >= ny) return;

    // ── 东/西面 (i+1/2, j) ──
    if (f == 0) {
        // 西边界: 直接拷贝或 BC 外插
        phi_f[j * (nx + 1) + f] = phi_c[j * nx + 0];
    } else if (f == nx) {
        // 东边界
        phi_f[j * (nx + 1) + f] = phi_c[j * nx + (nx - 1)];
    } else {
        int iL = f - 1;  // 左邻体心
        int iR = f;      // 右邻体心
        int idxL = j * nx + iL;
        int idxR = j * nx + iR;

        // Rhie-Chow 插值:
        // u_f = ū_f - D̄_f · (p_R - p_L - ∇p̄_f · d)
        // 其中 ū_f = 0.5·(u_L + u_R)
        //      D̄_f = 0.5·(D_L + D_R)
        //      ∇p̄_f · d = 0.5·(∇p_L + ∇p_R) · (x_R - x_L)
        float phi_avg = 0.5f * (phi_c[idxL] + phi_c[idxR]);
        float D_avg   = 0.5f * (Df[idxL] + Df[idxR]);
        float grad_avg_x = 0.5f * (grad_x[idxL] + grad_x[idxR]);

        // 压力梯度修正项: 去中心差分与线性插值的差
        float p_grad_term = (phi_c[idxR] - phi_c[idxL]) - grad_avg_x * dx;

        phi_f[j * (nx + 1) + f] = phi_avg - D_avg * p_grad_term;
    }
}


/* ═══════════════════════════════════════════════════════════════════
   核 2: Rhie-Chow 北/南面插值
   结构同上, 方向为 y
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_rhie_chow_faces_y(
    const float* __restrict__ phi_c,
    const float* __restrict__ grad_y,  // 体心 y 梯度
    const float* __restrict__ Df,
    float* phi_f,                      // 输出: 面值 [ny+1][nx]
    int nx, int ny,
    float dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int f = blockIdx.y * blockDim.y + threadIdx.y;  // 面索引沿 y
    if (i >= nx || f > ny) return;

    // ── 北/南面 (i, j+1/2) ──
    if (f == 0) {
        // 南边界
        phi_f[f * nx + i] = phi_c[0 * nx + i];
    } else if (f == ny) {
        // 北边界
        phi_f[f * nx + i] = phi_c[(ny - 1) * nx + i];
    } else {
        int jL = f - 1;
        int jR = f;
        int idxL = jL * nx + i;
        int idxR = jR * nx + i;

        float phi_avg = 0.5f * (phi_c[idxL] + phi_c[idxR]);
        float D_avg   = 0.5f * (Df[idxL] + Df[idxR]);
        float grad_avg_y = 0.5f * (grad_y[idxL] + grad_y[idxR]);

        float p_grad_term = (phi_c[idxR] - phi_c[idxL]) - grad_avg_y * dy;

        phi_f[f * nx + i] = phi_avg - D_avg * p_grad_term;
    }
}


/* ═══════════════════════════════════════════════════════════════════
   核 3: 通量系数 Df = cell_volume / aP 逐点计算
   为什么单独写个核? 因为 aP 来自动量方程, 可能在 CPU 端更新,
   需要把 Df 同步到 GPU. CuPy 直接 elementwise 也行, 但纯 CUDA 更透明.
   ─────────────────────────────────────────────────────────────────── */
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
