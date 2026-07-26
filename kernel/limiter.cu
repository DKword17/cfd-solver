/**
 * kernel/limiter.cu
 * TVD 限制器 CUDA 实现
 *
 * 高阶格式 (CDS, QUICK) 在间断处会产生非物理振荡.
 * TVD (Total Variation Diminishing) 限制器通过限制梯度
 * 在间断附近降阶为 1 阶迎风保证单调性.
 *
 * 这里实现三个常用限制器:
 *   minmod     — 最保守, 压缩性最强, 但耗散大
 *   van Leer   — 单调, 连续可微, 收敛性好
 *   superbee   — 最激进, Roe 推荐, 陡峭梯度分辨率高但可能"削尖"
 *
 * 每个限制器形式: φ(r) = limiter_function(r)
 * 其中 r = (Δ_upwind) / (Δ_local)
 *
 * 每个核: 256 线程/block (1D), 一次处理一个面的通量修正
 */

// ── device 函数: 三个 TVD 限制器 ──
// 为什么不用宏? 因为 device 函数能内联, 编译器可做常量传播优化
// 而且类型安全, 调试友好

// r: 相邻梯度比, r = (phi_C - phi_U) / (phi_D - phi_C)
// 其中 U=upwind, C=center, D=downwind
__device__ inline float _limiter_minmod(float r) {
    // minmod: φ(r) = max(0, min(1, r))
    return fmaxf(0.0f, fminf(1.0f, r));
}

__device__ inline float _limiter_van_leer(float r) {
    // van Leer: φ(r) = (r + |r|) / (1 + |r|)
    // 连续可微, 收敛性好
    float abs_r = fabsf(r);
    return (r + abs_r) / (1.0f + abs_r + 1e-15f);
}

__device__ inline float _limiter_superbee(float r) {
    // superbee: φ(r) = max(0, min(2r, 1), min(r, 2))
    // Roe 推荐, 陡峭梯度分辨率高
    return fmaxf(0.0f,
                 fmaxf(fminf(2.0f * r, 1.0f),
                       fminf(r, 2.0f)));
}


/* ═══════════════════════════════════════════════════════════════════
   核 1: x 方向 TVD 限制的面通量
   计算限制后的东面通量: phi_f = phi_C + 0.5·φ(r)·(phi_D - phi_C)

   输入:
     phi:      体心场 [ny][nx]
     limiter:  0=minmod, 1=vanLeer, 2=superbee
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_tvd_face_x(
    const float* __restrict__ phi,
    float* phi_f,        // 输出: x 方向面值 [ny][nx+1]
    int nx, int ny,
    int limiter_type     // 0=minmod, 1=vanLeer, 2=superbee
) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny) return;

    // 西边界
    phi_f[j * (nx + 1) + 0] = phi[j * nx + 0];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nx - 1;
         i += gridDim.x * blockDim.x) {
        int idxL = j * nx + i;      // φ_C (左)
        int idxR = j * nx + (i + 1); // φ_D (右)

        // 判断迎风方向: 假设 u > 0 (从左到右) 为基准
        // 实际使用时速度方向决定 upwind/downwind
        // 这里计算 r = (φ_C - φ_W) / (φ_D - φ_C)
        float phi_W = (i > 0) ? phi[j * nx + (i - 1)] : phi[idxL];
        float phi_C = phi[idxL];
        float phi_D = phi[idxR];

        float denom = phi_D - phi_C;
        float r = (phi_C - phi_W) / (denom + 1e-15f * (denom == 0.0f ? 0.0f : 1.0f));

        float limiter_val;
        switch (limiter_type) {
            case 0:  limiter_val = _limiter_minmod(r); break;
            case 1:  limiter_val = _limiter_van_leer(r); break;
            case 2:  limiter_val = _limiter_superbee(r); break;
            default: limiter_val = 1.0f; break;  // 无限制
        }

        // 限制后的面值: φ_f = φ_C + 0.5·φ(r)·(φ_D - φ_C)
        phi_f[j * (nx + 1) + (i + 1)] = phi_C + 0.5f * limiter_val * denom;
    }

    // 东边界
    phi_f[j * (nx + 1) + nx] = phi[j * nx + (nx - 1)];
}


/* ═══════════════════════════════════════════════════════════════════
   核 2: y 方向 TVD 限制的面通量
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_tvd_face_y(
    const float* __restrict__ phi,
    float* phi_f,        // 输出: y 方向面值 [ny+1][nx]
    int nx, int ny,
    int limiter_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;

    // 南边界
    phi_f[0 * nx + i] = phi[0 * nx + i];

    for (int j = blockIdx.y * blockDim.y + threadIdx.y;
         j < ny - 1;
         j += gridDim.y * blockDim.y) {
        int idxL = j * nx + i;      // φ_C (下)
        int idxR = (j + 1) * nx + i; // φ_D (上)

        float phi_S = (j > 0) ? (j - 1) * nx + i : idxL;
        float phi_C = phi[idxL];
        float phi_D = phi[idxR];

        float denom = phi_D - phi_C;
        float r = (phi_C - phi_S) / (denom + 1e-15f * (denom == 0.0f ? 0.0f : 1.0f));

        float limiter_val;
        switch (limiter_type) {
            case 0:  limiter_val = _limiter_minmod(r); break;
            case 1:  limiter_val = _limiter_van_leer(r); break;
            case 2:  limiter_val = _limiter_superbee(r); break;
            default: limiter_val = 1.0f; break;
        }

        phi_f[(j + 1) * nx + i] = phi_C + 0.5f * limiter_val * denom;
    }

    // 北边界
    phi_f[ny * nx + i] = phi[(ny - 1) * nx + i];
}


/* ═══════════════════════════════════════════════════════════════════
   核 3: 逐点限制器值查询
   用于 debug 或验证限制器行为
   ─────────────────────────────────────────────────────────────────── */
extern "C" __global__ void _kernel_limiter_value(
    const float* __restrict__ phi,
    float* r_field,          // 输出: 计算得到的 r 值
    float* phi_field,        // 输出: 限制器函数值
    int nx, int ny,
    int limiter_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;

    int idx = j * nx + i;

    // x 方向 r 值
    float phi_W = phi[j * nx + (i - 1)];
    float phi_C = phi[idx];
    float phi_E = phi[j * nx + (i + 1)];

    float denom = phi_E - phi_C;
    float r = (phi_C - phi_W) / (denom + 1e-15f);
    r_field[idx] = r;

    // 限制器函数值
    switch (limiter_type) {
        case 0:  phi_field[idx] = _limiter_minmod(r); break;
        case 1:  phi_field[idx] = _limiter_van_leer(r); break;
        case 2:  phi_field[idx] = _limiter_superbee(r); break;
        default: phi_field[idx] = 1.0f; break;
    }
}
