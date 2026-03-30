# 大气传输效应模拟工具

## 项目简介

本工具基于 Python 实现大气传输效应的模拟与可视化，主要功能包括：

1. **大气消光系数计算** — 基于 Beer-Lambert 定律，支持 Kruse 和 Kim 两种经验模型
2. **大气透过率计算** — 根据消光系数和传输距离计算大气透过率
3. **蒙特卡洛光子传输模拟** — 模拟大量光子在大气中的散射、吸收过程，生成**点扩散函数 (PSF)**
4. **图像卷积** — 将 PSF 与输入图像进行二维卷积，模拟大气传输效应对红外图像的退化影响
5. **可视化** — 生成 PSF 热力图、截面曲线、三维分布图、透过率曲线、图像对比图等

## 物理模型说明

### Beer-Lambert 定律

大气透过率 τ 与消光系数 β 和传输距离 R 的关系为：

```
τ = exp(-β × R)
```

### Kruse/Kim 消光模型

消光系数 β (km⁻¹) 基于大气能见度 V (km) 和工作波长 λ (μm) 计算：

```
β = (3.912 / V) × (λ / 0.55)^(-q)
```

其中 q 为波长依赖指数，由 Kruse 模型或 Kim 模型确定：

| 模型 | 能见度范围 | q 值 |
|------|-----------|------|
| **Kruse** | V > 50 km | 1.6 |
| | 6 km ≤ V ≤ 50 km | 1.3 |
| | V < 6 km | 0.585 × V^(1/3) |
| **Kim** | V > 50 km | 1.6 |
| | 6 km < V ≤ 50 km | 1.3 |
| | 1 km < V ≤ 6 km | 0.16V + 0.34 |
| | 0.5 km < V ≤ 1 km | V - 0.5 |
| | V ≤ 0.5 km | 0 |

### 蒙特卡洛光子传输

模拟步骤：

1. 从点光源沿 z 轴方向发射大量光子
2. 对每个光子，随机采样自由程（两次碰撞间的距离）
3. 在碰撞点根据单次散射反照率 ω₀ 判断光子是被吸收还是被散射
4. 若散射，基于 **Henyey-Greenstein 相函数**采样新的散射方向
5. 重复步骤 2-4，直到光子到达成像平面或被吸收
6. 统计所有到达成像平面的光子落点分布，归一化后得到 PSF

### 图像卷积

最终退化图像由原始图像与 PSF 的二维卷积得到：

```
I_final(x, y) = I_original(x, y) ⊗ PSF(x, y)
```

## 安装

### 环境要求

- Python >= 3.10

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：
- `numpy` — 数值计算
- `scipy` — 卷积运算
- `matplotlib` — 可视化
- `Pillow` — 图像读写

## 使用方法

### 命令行方式

```bash
# 仅生成 PSF 及可视化（不处理图像）
python main.py --visibility 10 --distance 5

# 对已有图像施加大气传输效应
python main.py --visibility 5 --distance 3 --image input.png --output output

# 使用 Kruse 模型，自定义光子数
python main.py --visibility 10 --distance 5 --model kruse --photons 200000

# 低能见度（浓雾）场景
python main.py --visibility 1 --distance 2 --photons 500000 --psf-size 151
```

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--visibility` | 大气能见度 (km) | 10.0 |
| `--distance` | 传输距离 (km) | 5.0 |
| `--wavelength` | 工作波长 (μm) | 10.0 |
| `--model` | 消光模型 (kruse/kim) | kim |
| `--photons` | 蒙特卡洛光子数 | 100000 |
| `--asymmetry` | Henyey-Greenstein 不对称因子 g | 0.85 |
| `--scattering-ratio` | 单次散射反照率 ω₀ | 0.9 |
| `--psf-size` | PSF 矩阵边长（像素） | 101 |
| `--pixel-scale` | 像素对应物理尺寸 (km) | 0.01 |
| `--seed` | 随机数种子 | 42 |
| `--image` | 输入图像路径 | 无 |
| `--output` | 输出目录 | output |

### Python API 方式

```python
from src.atmospheric_transport import (
    extinction_coefficient,
    atmospheric_transmittance,
    monte_carlo_psf,
    apply_psf,
    simulate,
)
from src.visualization import (
    plot_psf_2d,
    plot_psf_cross_section,
    plot_psf_3d,
    plot_transmittance_curve,
    plot_image_comparison,
)
import numpy as np
from PIL import Image

# ── 方式一：一站式模拟 ──────────────────────────────
result = simulate(
    visibility_km=5.0,      # 能见度 5 km
    distance_km=3.0,         # 传输距离 3 km
    wavelength_um=10.0,      # 长波红外 10 μm
    model="kim",             # Kim 消光模型
    n_photons=100_000,       # 10 万光子
    asymmetry_factor=0.85,   # 前向散射为主
    scattering_ratio=0.9,    # 单次散射反照率
    psf_size=101,            # PSF 尺寸 101×101
    seed=42,
)

print(f"消光系数: {result['extinction_coeff']:.4f} km⁻¹")
print(f"大气透过率: {result['transmittance']:.4f}")
psf = result["psf"]

# ── 方式二：分步调用 ─────────────────────────────────
# 1. 计算消光系数
beta = extinction_coefficient(visibility_km=5.0, wavelength_um=10.0, model="kim")

# 2. 计算透过率
tau = atmospheric_transmittance(visibility_km=5.0, distance_km=3.0)

# 3. 蒙特卡洛生成 PSF
psf = monte_carlo_psf(
    n_photons=100_000,
    extinction_coeff=beta,
    scattering_ratio=0.9,
    distance_km=3.0,
    asymmetry_factor=0.85,
    psf_size=101,
    seed=42,
)

# 4. 对图像施加 PSF（卷积）
image = np.array(Image.open("input.png"))
degraded = apply_psf(image, psf)
Image.fromarray(degraded).save("degraded.png")

# ── 可视化 ───────────────────────────────────────────
plot_psf_2d(psf, save_path="psf_2d.png")
plot_psf_cross_section(psf, save_path="psf_cross.png")
plot_psf_3d(psf, save_path="psf_3d.png")
plot_transmittance_curve(save_path="transmittance.png")
plot_image_comparison(image, degraded, save_path="comparison.png")
```

## 输出说明

运行后在指定输出目录中生成以下文件：

| 文件名 | 说明 |
|--------|------|
| `psf_2d.png` | PSF 二维热力图（对数色标） |
| `psf_cross_section.png` | PSF 中心行/列截面曲线 |
| `psf_3d.png` | PSF 三维表面图 |
| `transmittance_curve.png` | 不同能见度下大气透过率随距离变化曲线 |
| `psf_data.npy` | PSF 原始数据（numpy 格式，可加载复用） |
| `degraded_image.png` | 经大气传输效应退化后的图像（需指定 `--image`） |
| `comparison.png` | 原始图像与退化图像并排对比（需指定 `--image`） |

## 运行测试

```bash
pip install pytest
python -m pytest tests/ -v
```

## 项目结构

```
Atmospheric-transport/
├── main.py                              # 命令行入口
├── requirements.txt                     # Python 依赖
├── README.md                            # 本文档
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── atmospheric_transport.py         # 核心模块（消光、透过率、蒙特卡洛、卷积）
│   └── visualization.py                # 可视化模块
├── tests/
│   ├── __init__.py
│   └── test_atmospheric_transport.py   # 单元测试
└── 曹疏桐-开题报告.pdf                   # 原始开题报告
```

## 参考文献

- Kruse P W, McGlauchlin L D, McQuistan R B. *Elements of Infrared Technology: Generation, Transmission and Detection*. 1962.
- Kim I I, McArthur B, Korevaar E J. Comparison of laser beam propagation at 785 nm and 1550 nm in fog and haze for optical wireless communications. *Proc. SPIE*, 2001.
- Henyey L G, Greenstein J L. Diffuse radiation in the Galaxy. *Astrophysical Journal*, 1941, 93: 70-83.
