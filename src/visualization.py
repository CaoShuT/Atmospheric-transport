"""大气传输效应可视化模块。

提供 PSF 可视化、透过率曲线、图像对比等绘图功能。
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_LOG_SCALE_MIN = 1e-10   # 对数色标最小值
_LOG_PLOT_FLOOR = 1e-30  # 三维图对数变换下限


def plot_psf_2d(
    psf: np.ndarray,
    title: str = "大气点扩散函数 (PSF)",
    save_path: str | None = None,
    log_scale: bool = True,
) -> plt.Figure:
    """绘制 PSF 的二维热力图。

    Parameters
    ----------
    psf : numpy.ndarray
        PSF 矩阵。
    title : str
        图像标题。
    save_path : str or None
        保存路径。如为 None 则不保存。
    log_scale : bool
        是否使用对数色标。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图像对象。
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    psf_display = psf.copy()
    if log_scale:
        psf_display = np.where(psf_display > 0, psf_display, np.nan)
        vmin = np.nanmin(psf_display[psf_display > 0]) if np.any(psf_display > 0) else _LOG_SCALE_MIN
        im = ax.imshow(psf_display, cmap="hot", norm=LogNorm(vmin=vmin))
    else:
        im = ax.imshow(psf_display, cmap="hot")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("像素 x")
    ax.set_ylabel("像素 y")
    plt.colorbar(im, ax=ax, label="归一化能量")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_psf_cross_section(
    psf: np.ndarray,
    title: str = "PSF 截面分布",
    save_path: str | None = None,
) -> plt.Figure:
    """绘制 PSF 中心行和列的截面曲线。

    Parameters
    ----------
    psf : numpy.ndarray
        PSF 矩阵。
    title : str
        图像标题。
    save_path : str or None
        保存路径。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图像对象。
    """
    center = psf.shape[0] // 2
    row_profile = psf[center, :]
    col_profile = psf[:, center]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(row_profile, "r-", linewidth=1.5)
    ax1.set_title("水平截面 (中心行)")
    ax1.set_xlabel("像素位置")
    ax1.set_ylabel("归一化能量")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    ax2.plot(col_profile, "b-", linewidth=1.5)
    ax2.set_title("垂直截面 (中心列)")
    ax2.set_xlabel("像素位置")
    ax2.set_ylabel("归一化能量")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_psf_3d(
    psf: np.ndarray,
    title: str = "PSF 三维分布",
    save_path: str | None = None,
) -> plt.Figure:
    """绘制 PSF 的三维表面图。

    Parameters
    ----------
    psf : numpy.ndarray
        PSF 矩阵。
    title : str
        图像标题。
    save_path : str or None
        保存路径。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图像对象。
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    h, w = psf.shape
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    psf_log = np.log10(np.where(psf > 0, psf, _LOG_PLOT_FLOOR))

    ax.plot_surface(X, Y, psf_log, cmap="hot", linewidth=0, antialiased=True)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("像素 x")
    ax.set_ylabel("像素 y")
    ax.set_zlabel("log₁₀(能量)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_transmittance_curve(
    visibility_values: list[float] | None = None,
    distance_range: tuple[float, float] = (0.1, 20.0),
    wavelength_um: float = 10.0,
    model: str = "kim",
    n_points: int = 200,
    title: str = "大气透过率随传输距离变化曲线",
    save_path: str | None = None,
) -> plt.Figure:
    """绘制不同能见度下透过率随距离的变化曲线。

    Parameters
    ----------
    visibility_values : list[float] or None
        要绘制的能见度列表 (km)。默认使用 [1, 3, 5, 10, 23]。
    distance_range : tuple[float, float]
        距离范围 (km)。
    wavelength_um : float
        工作波长 (μm)。
    model : str
        消光模型。
    n_points : int
        曲线采样点数。
    title : str
        图像标题。
    save_path : str or None
        保存路径。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图像对象。
    """
    from .atmospheric_transport import atmospheric_transmittance

    if visibility_values is None:
        visibility_values = [1.0, 3.0, 5.0, 10.0, 23.0]

    distances = np.linspace(distance_range[0], distance_range[1], n_points)

    fig, ax = plt.subplots(figsize=(10, 6))

    for vis in visibility_values:
        tau_values = [
            atmospheric_transmittance(vis, d, wavelength_um, model) for d in distances
        ]
        ax.plot(distances, tau_values, linewidth=2, label=f"V = {vis} km")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("传输距离 (km)", fontsize=12)
    ax.set_ylabel("大气透过率 τ", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_image_comparison(
    original: np.ndarray,
    degraded: np.ndarray,
    title: str = "大气传输效应前后对比",
    save_path: str | None = None,
) -> plt.Figure:
    """并排显示原始图像与经大气传输效应退化后的图像。

    Parameters
    ----------
    original : numpy.ndarray
        原始图像。
    degraded : numpy.ndarray
        退化后图像。
    title : str
        图像标题。
    save_path : str or None
        保存路径。

    Returns
    -------
    matplotlib.figure.Figure
        生成的图像对象。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap = "gray" if original.ndim == 2 else None
    ax1.imshow(original, cmap=cmap)
    ax1.set_title("原始图像", fontsize=13)
    ax1.axis("off")

    ax2.imshow(degraded, cmap=cmap)
    ax2.set_title("大气传输效应退化图像", fontsize=13)
    ax2.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
