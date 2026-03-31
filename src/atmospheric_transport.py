"""大气传输效应模拟核心模块。

实现基于 Beer-Lambert 定律、Kruse/Kim 消光模型和蒙特卡洛方法的
大气传输效应模拟，生成点扩散函数(PSF)并对图像进行卷积处理。
"""

import numpy as np
from scipy.signal import fftconvolve

# 数值常量
_G_EPSILON = 1e-12          # 不对称因子接近零的阈值
_MIN_RANDOM_VALUE = 1e-30   # 避免 log(0) 的随机数下限
_DZ_THRESHOLD = 0.99999     # 方向向量 z 分量退化阈值


# ────────────────────────────────────────────────────────────────────
# 1. 消光系数模型 (Extinction Coefficient Models)
# ────────────────────────────────────────────────────────────────────


def _kruse_q(visibility_km: float) -> float:
    """计算 Kruse 模型中的波长依赖指数 q。

    Parameters
    ----------
    visibility_km : float
        大气能见度 (km)。

    Returns
    -------
    float
        波长依赖指数 q。
    """
    if visibility_km > 50.0:
        return 1.6
    if visibility_km >= 6.0:
        return 1.3
    return 0.585 * visibility_km ** (1.0 / 3.0)


def _kim_q(visibility_km: float) -> float:
    """计算 Kim 模型中的波长依赖指数 q。

    Parameters
    ----------
    visibility_km : float
        大气能见度 (km)。

    Returns
    -------
    float
        波长依赖指数 q。
    """
    if visibility_km > 50.0:
        return 1.6
    if visibility_km > 6.0:
        return 1.3
    if visibility_km > 1.0:
        return 0.16 * visibility_km + 0.34
    if visibility_km > 0.5:
        return visibility_km - 0.5
    return 0.0


def extinction_coefficient(
    visibility_km: float,
    wavelength_um: float = 10.0,
    model: str = "kim",
) -> float:
    """计算大气消光系数 (km⁻¹)。

    基于 Beer-Lambert 定律和 Kruse/Kim 经验模型，利用大气能见度和
    工作波长计算消光系数。

    Parameters
    ----------
    visibility_km : float
        大气能见度 (km)，必须大于 0。
    wavelength_um : float
        工作波长 (μm)，默认为 10 μm（长波红外）。
    model : str
        经验模型选择，``"kruse"`` 或 ``"kim"``（默认）。

    Returns
    -------
    float
        消光系数 β_ext (km⁻¹)。

    Raises
    ------
    ValueError
        当 visibility_km <= 0 或 model 不被支持时。
    """
    if visibility_km <= 0:
        raise ValueError("能见度必须大于 0")

    model = model.lower()
    if model == "kruse":
        q = _kruse_q(visibility_km)
    elif model == "kim":
        q = _kim_q(visibility_km)
    else:
        raise ValueError(f"不支持的模型: {model}，请选择 'kruse' 或 'kim'")

    # 参考波长 0.55 μm（人眼最大灵敏度波长）
    lambda_ref = 0.55
    # Beer-Lambert: β_ext = (3.912 / V) * (λ / 0.55)^(-q)
    beta_ext = (3.912 / visibility_km) * (wavelength_um / lambda_ref) ** (-q)
    return beta_ext


def atmospheric_transmittance(
    visibility_km: float,
    distance_km: float,
    wavelength_um: float = 10.0,
    model: str = "kim",
) -> float:
    """计算大气透过率。

    基于 Beer-Lambert 定律: τ = exp(-β_ext * R)

    Parameters
    ----------
    visibility_km : float
        大气能见度 (km)。
    distance_km : float
        传输距离 (km)，必须 >= 0。
    wavelength_um : float
        工作波长 (μm)。
    model : str
        消光模型。

    Returns
    -------
    float
        大气透过率 τ ∈ [0, 1]。

    Raises
    ------
    ValueError
        当 distance_km < 0 时。
    """
    if distance_km < 0:
        raise ValueError("传输距离不能为负数")
    if distance_km == 0:
        return 1.0

    beta = extinction_coefficient(visibility_km, wavelength_um, model)
    tau = np.exp(-beta * distance_km)
    return float(tau)


# ────────────────────────────────────────────────────────────────────
# 2. 蒙特卡洛光子传输模拟 (Monte Carlo Photon Transport)
# ────────────────────────────────────────────────────────────────────


def _henyey_greenstein_scatter_angle(g: float, rng: np.random.Generator) -> float:
    """基于 Henyey-Greenstein 相函数采样散射角 θ。

    Parameters
    ----------
    g : float
        不对称因子，范围 [-1, 1]。g > 0 表示前向散射为主。
    rng : numpy.random.Generator
        随机数生成器。

    Returns
    -------
    float
        散射角 θ (rad)。
    """
    xi = rng.random()
    if abs(g) < _G_EPSILON:
        cos_theta = 1.0 - 2.0 * xi
    else:
        s = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
        cos_theta = (1.0 + g * g - s * s) / (2.0 * g)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def monte_carlo_psf(
    n_photons: int = 100_000,
    extinction_coeff: float = 1.0,
    scattering_ratio: float = 0.9,
    distance_km: float = 5.0,
    asymmetry_factor: float = 0.85,
    psf_size: int = 101,
    pixel_scale: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """通过蒙特卡洛方法模拟大气传输并生成点扩散函数 (PSF)。

    模拟大量光子从点光源出发、经过大气散射和吸收后到达成像平面的
    过程，统计光子落点分布以获得 PSF。

    Parameters
    ----------
    n_photons : int
        模拟光子总数。
    extinction_coeff : float
        消光系数 β_ext (km⁻¹)。
    scattering_ratio : float
        单次散射反照率 ω₀ = β_scat / β_ext，范围 (0, 1]。
    distance_km : float
        光源到成像平面的距离 (km)。
    asymmetry_factor : float
        Henyey-Greenstein 不对称因子 g，范围 [-1, 1]。
    psf_size : int
        PSF 矩阵边长（像素数），必须为奇数。
    pixel_scale : float
        每个像素对应的物理尺寸 (km)。
    seed : int or None
        随机数种子，用于结果复现。

    Returns
    -------
    numpy.ndarray
        归一化 PSF 矩阵，形状 (psf_size, psf_size)，总和为 1。
    """
    if psf_size % 2 == 0:
        psf_size += 1

    rng = np.random.default_rng(seed)
    psf = np.zeros((psf_size, psf_size), dtype=np.float64)
    center = psf_size // 2

    absorption_coeff = extinction_coeff * (1.0 - scattering_ratio)
    scattering_coeff = extinction_coeff * scattering_ratio

    for _ in range(n_photons):
        # 光子初始状态：从原点出发沿 z 轴正方向
        x, y, z = 0.0, 0.0, 0.0
        dx, dy, dz = 0.0, 0.0, 1.0
        weight = 1.0
        alive = True

        while alive and z < distance_km:
            # 采样自由程
            xi = rng.random()
            if xi < _MIN_RANDOM_VALUE:
                xi = _MIN_RANDOM_VALUE
            step = -np.log(xi) / extinction_coeff

            # 移动光子
            x += dx * step
            y += dy * step
            z += dz * step

            if z >= distance_km:
                # 光子到达成像平面，按剩余比例投影
                overshoot = z - distance_km
                x -= dx * overshoot
                y -= dy * overshoot
                z = distance_km
                break

            # 判断吸收/散射
            xi1 = rng.random()
            if xi1 <= absorption_coeff / extinction_coeff:
                alive = False  # 光子被吸收
            else:
                # 散射：采样新方向
                theta = _henyey_greenstein_scatter_angle(asymmetry_factor, rng)
                phi = 2.0 * np.pi * rng.random()

                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                # 旋转方向向量
                if abs(dz) > _DZ_THRESHOLD:
                    new_dx = sin_theta * cos_phi
                    new_dy = sin_theta * sin_phi
                    new_dz = cos_theta * np.sign(dz)
                else:
                    denom = np.sqrt(1.0 - dz * dz)
                    new_dx = (
                        sin_theta * (dx * dz * cos_phi - dy * sin_phi) / denom
                        + dx * cos_theta
                    )
                    new_dy = (
                        sin_theta * (dy * dz * cos_phi + dx * sin_phi) / denom
                        + dy * cos_theta
                    )
                    new_dz = -sin_theta * cos_phi * denom + dz * cos_theta

                norm = np.sqrt(new_dx**2 + new_dy**2 + new_dz**2)
                dx, dy, dz = new_dx / norm, new_dy / norm, new_dz / norm

        if alive and z >= distance_km:
            # 映射到像素坐标
            px = int(round(x / pixel_scale)) + center
            py = int(round(y / pixel_scale)) + center
            if 0 <= px < psf_size and 0 <= py < psf_size:
                psf[py, px] += weight

    # 归一化
    total = psf.sum()
    if total > 0:
        psf /= total

    return psf


# ────────────────────────────────────────────────────────────────────
# 3. 图像卷积 (Image Convolution)
# ────────────────────────────────────────────────────────────────────


def apply_psf(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """将 PSF 与输入图像进行二维卷积，模拟大气传输效应。

    Ifinal(x, y) = Ioriginal(x, y) ⊗ PSF(x, y)

    Parameters
    ----------
    image : numpy.ndarray
        输入图像，灰度图 (H, W) 或彩色图 (H, W, C)。
    psf : numpy.ndarray
        点扩散函数矩阵 (M, M)。

    Returns
    -------
    numpy.ndarray
        卷积后的图像，与输入图像具有相同的形状和数据范围。
    """
    if image.ndim == 2:
        result = fftconvolve(image.astype(np.float64), psf, mode="same")
    elif image.ndim == 3:
        result = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            result[:, :, c] = fftconvolve(
                image[:, :, c].astype(np.float64), psf, mode="same"
            )
    else:
        raise ValueError(f"不支持的图像维度: {image.ndim}，需要 2 或 3")

    # 裁剪到原始数据范围
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        result = np.clip(result, info.min, info.max)
        return result.astype(image.dtype)
    else:
        return result.astype(image.dtype)


# ────────────────────────────────────────────────────────────────────
# 4. 便捷接口 (Convenience API)
# ────────────────────────────────────────────────────────────────────


def simulate(
    visibility_km: float = 10.0,
    distance_km: float = 5.0,
    wavelength_um: float = 10.0,
    model: str = "kim",
    n_photons: int = 100_000,
    asymmetry_factor: float = 0.85,
    scattering_ratio: float = 0.9,
    psf_size: int = 101,
    pixel_scale: float = 0.01,
    seed: int | None = 42,
) -> dict:
    """一站式大气传输效应模拟。

    依次执行：计算消光系数 → 计算透过率 → 蒙特卡洛生成 PSF。

    Parameters
    ----------
    visibility_km : float
        大气能见度 (km)。
    distance_km : float
        传输距离 (km)。
    wavelength_um : float
        工作波长 (μm)。
    model : str
        消光模型 (``"kruse"`` / ``"kim"``)。
    n_photons : int
        模拟光子总数。
    asymmetry_factor : float
        Henyey-Greenstein 不对称因子。
    scattering_ratio : float
        单次散射反照率。
    psf_size : int
        PSF 矩阵边长。
    pixel_scale : float
        像素尺度 (km)。
    seed : int or None
        随机数种子。

    Returns
    -------
    dict
        包含以下键值:
        - ``"extinction_coeff"``: 消光系数 (km⁻¹)
        - ``"transmittance"``: 大气透过率
        - ``"psf"``: PSF 矩阵 (numpy.ndarray)
        - ``"params"``: 输入参数字典
    """
    beta = extinction_coefficient(visibility_km, wavelength_um, model)
    tau = atmospheric_transmittance(visibility_km, distance_km, wavelength_um, model)
    psf = monte_carlo_psf(
        n_photons=n_photons,
        extinction_coeff=beta,
        scattering_ratio=scattering_ratio,
        distance_km=distance_km,
        asymmetry_factor=asymmetry_factor,
        psf_size=psf_size,
        pixel_scale=pixel_scale,
        seed=seed,
    )

    return {
        "extinction_coeff": beta,
        "transmittance": tau,
        "psf": psf,
        "params": {
            "visibility_km": visibility_km,
            "distance_km": distance_km,
            "wavelength_um": wavelength_um,
            "model": model,
            "n_photons": n_photons,
            "asymmetry_factor": asymmetry_factor,
            "scattering_ratio": scattering_ratio,
            "psf_size": psf_size,
            "pixel_scale": pixel_scale,
        },
    }
