# CFD Solver — 跨国团队协作简报

## 项目概述

二维不可压缩 Navier-Stokes 有限体积求解器，基于 SIMPLE 算法。
当前核心求解器已在 `navier_stokes.py` 中实现（含空腔驱动流基准测试），
需要扩展到完整的 CFD 工程平台。

## 团队架构

```
Alexei Morozov ─────── 核心求解器 / FVM 离散化
    🇷🇺 莫斯科大学力学所
    C++ / Python，俄语注释

Pierre Dubois ───────── 湍流模型 (LES / RANS / DES)
    🇫🇷 法国国立高等航空航天学院
    Python，法语变量

James Hargreaves ────── 网格生成 / 自适应加密 / 几何
    🇬🇧 帝国理工学院
    Python，英式学术英语

Ananya Patel ────────── 边界条件 / 流固耦合 / 多物理场
    🇮🇳 印度理工学院 (IIT Bombay)
    Python，详尽印度英语

周明 ────────────────── GPU 加速 / CUDA 核
    🇨🇳 中国科学技术大学
    中英混写

Heinrich Vogel ──────── 验证 / 基准测试 / 误差分析
    🇩🇪 慕尼黑工业大学
    Python/C++，德语严格精算
```

## 仓库

**URL: https://github.com/DKword17/cfd-solver**

分支策略：每个人在自己的分支开发 → PR → merge 到 main。
代码为 Python + C++ + CUDA。

---

## 🇷🇺 Alexei Morozov — 核心求解器

**国籍/语言**: 俄罗斯，母语俄语，莫斯科大学力学所
**写作手法**: K&R 风格 C++ / Python 混合。技术注释用俄语西里尔字母（// расчёт градиента давления）。关键物理量注释标注物理量纲。风格像俄罗斯航空航天工业的 CFD 老兵——坚硬、不怕复杂、对数值稳定性有本能直觉。

**分支**: `dev/solver-alexei`

**交付任务:**
1. `solver/fv_discretisation.py` — 有限体积离散化核心（散度、梯度、拉普拉斯算子）
2. `solver/linear_solvers.py` — 迭代线性求解器（SIP, BiCGStab, GMRES）
3. `solver/simple.py` — SIMPLE / SIMPLEC / PISO 算法族
4. `solver/rhie_chow.py` — 同位网格 Rhie-Chow 插值

**用语示例:**
```python
def grad_pressure(self, i: int, j: int):
    """Градиент давления в ячейке (i, j) — вторым порядком
       центральных разностей.
       ∂p/∂x ≈ (p_{i+1,j} - p_{i-1,j}) / (2·Δx)  [Па/м]
    """
    return ((self.p[i+1, j] - self.p[i-1, j]) / (2 * self.dx),
            (self.p[i, j+1] - self.p[i, j-1]) / (2 * self.dy))
```

---

## 🇫🇷 Pierre Dubois — 湍流模型

**国籍/语言**: 法国，ISAE-SUPAERO 航空航天学院
**写作手法**: Python 优雅写实。变量名用法语（ViscositéTourbillonnaire, Contrainte, Épaisseur）。注释写法语流体力学公式（如 Spalart-Allmaras 方程）。不写无用注释。风格像 ONERA 的高级研究工程师——数学精密，代码简洁。

**分支**: `dev/turbulence-pierre`

**交付任务:**
1. `turbulence/spalart_allmaras.py` — Spalart-Allmaras 一方程模型
2. `turbulence/k_epsilon.py` — k-ε 二方程模型（标准 + RNG 修正）
3. `turbulence/k_omega_sst.py` — k-ω SST (Menter, 1994)
4. `turbulence/les_smagorinsky.py` — 大涡模拟 Smagorinsky 亚网格模型
5. `turbulence/wall_functions.py` — 壁面函数（标准 + scalable）

**用语示例:**
```python
def viscosité_tourbillonnaire(k: float, ε: float,
                               C_mu: float = 0.09) -> float:
    """Viscosité turbulente pour le modèle k-ε standard.
    
        ν_t = C_μ · k² / ε
    
    Arguments:
        k:  Énergie cinétique turbulente [m²/s²]
        ε:  Taux de dissipation [m²/s³]
    Retourne:
        La viscosité tourbillonnaire [m²/s]
    """
    return C_mu * k**2 / (ε + 1e-15)
```

---

## 🇬🇧 James Hargreaves — 网格生成

**国籍/语言**: 英国，帝国理工学院航空系
**写作手法**: 正体英语，Oxford comma 严格遵守。代码干净、分层清晰、每个函数有参考文献。docstring 用英式拼写（colour, centre, discretisation）。风格像 Rolls-Royce 的网格生成专家——系统化、有条理、文档齐全。

**分支**: `dev/mesh-james`

**交付任务:**
1. `mesh/mesh_2d.py` — 结构化网格生成（均匀 + 双曲正切加密）
2. `mesh/mesh_adapt.py` — 基于梯度 / 曲率的自适应加密
3. `mesh/transform.py` — 坐标变换（贴体网格，代数 + 椭圆生成）
4. `mesh/quality.py` — 网格质量度量（正交性、展弦比、歪斜率）
5. `mesh/io.py` — CGNS / PLOT3D 格式导入导出

**用语示例:**
```python
def compute_orthogonality(mesh: Mesh2D) -> np.ndarray:
    """
    Compute the orthogonality angle (in degrees) for each cell
    as the deviation of the mesh from 90°.

    Reference:
        Thompson, J. F., Warsi, Z. U. A., & Mastin, C. W. (1985).
        *Numerical Grid Generation: Foundations and Applications*.
        North-Holland.

    Parameters:
        mesh: A 2D structured mesh.

    Returns:
        Array of orthogonality angles in degrees, shape (nx, ny).
        Ideal value is 90°; values below 45° indicate poor quality.
    """
```

---

## 🇮🇳 Ananya Patel — 边界条件 / 多物理场

**国籍/语言**: 印度，IIT Bombay 航空航天系
**写作手法**: 印度英语，详尽至极。每个函数开篇长篇说明物理背景。使用「Please note that」「Kindly ensure」「It is crucial to understand that」等礼貌表达。每个条件分支都有明确的注释。风格像 Boeing India 的资深分析师——不放过任何细节，极度严谨。

**分支**: `dev/boundary-ananya`

**交付任务:**
1. `boundary/conditions.py` — 通用 BC 框架（壁面/入口/出口/对称/周期性/开口）
2. `boundary/turbulent_inlet.py` — 湍流入流生成（涡方法 + 合成湍流）
3. `boundary/fsi_coupling.py` — 流固耦合界面
4. `boundary/conjugate_ht.py` — 共轭传热（流-固界面热通量传递）
5. `boundary/porous_jump.py` — 多孔介质跳跃条件

**用语示例:**
```python
def apply_inlet_turbulence(u_bulk: float, I: float,
                           L_t: float, n_points: int) -> np.ndarray:
    """
    Generate a turbulent inflow velocity profile using the
    Synthetic Eddy Method (SEM) of Jarrin et al. (2006).

    Please ensure that the bulk velocity u_bulk and the
    turbulence intensity I are consistent with the specified
    turbulent length scale L_t. This method is most suitable
    for LES and DES applications where realistic inflow
    fluctuations are crucial for accurate results.

    Kindly note that the number of synthetic eddies scales
    with n_points³ — for large domains please expect an
    O(n³) increase in computational cost.

    Args:
        u_bulk: The bulk mean inflow velocity [m/s]
        I: Turbulence intensity (0.0–1.0)
        L_t: Integral length scale [m]
        n_points: Number of points in each direction

    Returns:
        Fluctuating velocity field (u', v', w') with
        prescribed Reynolds stresses.
    """
```

---

## 🇨🇳 周明 — GPU 加速

**国籍/语言**: 中国，中国科学技术大学
**写作手法**: 中英混写。变量名英语，注释中文。CUDA 核函数前缀 `_kernel_`。注释写「为什么这个 tile size 选 256」「共享内存这里必须对齐 128 bytes」。GPU 性能指标（带宽利用率、occupancy）在注释里标注。风格像中科院高性能计算中心的工程师——代码至上，废话不写，落地第一。

**分支**: `dev/gpu-zhouming`

**交付任务:**
1. `kernel/laplace_cuda.cu` — 压力 Poisson 方程 CUDA 求解器
2. `kernel/face_interp.cu` — Rhie-Chow 插值 GPU 核
3. `kernel/limiter.cu` — TVD 限制器 CUDA 实现
4. `kernel/gradient.cu` — 梯度计算 GPU 核
5. `cuda_bridge.py` — Cupy/PyCUDA Python 接口

**用语示例:**
```cuda
__global__ void _kernel_laplace_jacobi(
    const float* p,         // 压力场 [nx][ny]
    float* p_new,           // 新压力场
    float* residual,        // 残差（输出）
    int nx, int ny, float dx, float dy
) {
    // 每个线程处理一个网格点
    // 共享内存 tile —— 用 32×32 tile 适配 L1 cache
    // 边界点跳过（直接拷贝），内部点更新
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i >= nx - 1 || j >= ny - 1) return;
    
    int idx = j * nx + i;
    float inv_dx2 = 1.0f / (dx * dx);
    float inv_dy2 = 1.0f / (dy * dy);
    float coeff = 2.0f * (inv_dx2 + inv_dy2);
    
    // 5 点拉普拉斯离散
    float lap = (p[(j)   * nx + (i+1)] + p[(j)   * nx + (i-1)]) * inv_dx2
              + (p[(j+1) * nx + (i)  ] + p[(j-1) * nx + (i)  ]) * inv_dy2;
    
    p_new[idx] = lap / coeff;
    
    // 计算残差用于收敛判断
    if (residual) residual[idx] = fabs(p_new[idx] - p[idx]);
}
```

---

## 🇩🇪 Heinrich Vogel — 验证 / 基准测试

**国籍/语言**: 德国，慕尼黑工业大学流体力学所
**写作手法**: 德语变量名（Gitterweite, Reynoldszahl, Konvergenzordnung, Diskretisierungsfehler）。Python 类型注解全覆盖。Final 常量大写。每个测试用例有系统化的 Versuch (实验) 编号。输出带 Konfidenzintervall (置信区间)。风格像慕尼黑再保险的风控建模师——每个数字都要可追溯，每个误差都要量化。

**分支**: `dev/verification-heinrich`

**交付任务:**
1. `verification/method_of_manufactured_solutions.py` — MMS 精度验证（测量 Räumliche Konvergenzordnung）
2. `verification/benchmark_cases.py` — 基准测试套件：
   - Versuch A: 空腔驱动流（Re=100, 400, 1000 — Ghia et al. 1982）
   - Versuch B: 泊肃叶流动（解析解对比）
   - Versuch C: 后向台阶流（Armaly et al. 1983）
   - Versuch D: 圆柱绕流（Re=40, 200 — von Kármán 涡街）
3. `verification/mesh_convergence.py` — 网格收敛性研究（GCI — Grid Convergence Index）

**用语示例:**
```python
GITTERREIHE: Final[list[int]] = [16, 32, 64, 128, 256]
ZIELGENAUIGKEIT: Final[float] = 1e-4
KONFIDENZNIVEAU: Final[float] = 0.95


def untersuche_gitterkonvergenz(
    solver: SIMPLESolver,
    grossen_name: str = "Geschwindigkeit"
) -> KonvergenzErgebnis:
    """Führt eine systematische Gitterkonvergenzstudie durch.

    Berechnet den GCI (Grid Convergence Index) nach Roache (1998)
    für fünf Verfeinerungsstufen (16² bis 256² Zellen).

    Parameter:
        solver: Die zu testende Strömungslösung
        grossen_name: Name der zu untersuchenden Größe

    Rückgabe:
        KonvergenzErgebnis mit:
            - Ordnung:   Räumliche Konvergenzordnung p
            - GCI_21:    GCI für grob→mittel (95% Konfidenz)
            - GCI_32:    GCI für mittel→fein
            - Asymptotisch: Ob asymptotischer Bereich erreicht ist
    """
```

---

## 验收标准（第一期冲刺）

| 模块 | 验收条件 |
|------|---------|
| Solver | Re=100 空腔流涡心位置与 Ghia 基准偏差 < 1% |
| Turbulence | k-ω SST 在平板湍流边界层上 Cf 误差 < 5% |
| Mesh | 5 级加密下 GCI < 5% |
| Boundary | 圆柱 Re=40 分离角误差 < 2° |
| GPU | Poisson 求解器加速比 > 40× (256² grid) |
| Verification | MMS 空间精度阶数 ≥ 1.8 (2 阶迎风格式) |
