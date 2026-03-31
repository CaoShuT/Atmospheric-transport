"""大气传输效应模拟 - 命令行入口。

用法示例:
    # 仅生成 PSF 可视化
    python main.py --visibility 5 --distance 3

    # 对已有图像施加大气传输效应
    python main.py --visibility 5 --distance 3 --image input.png --output output

    # 使用 Kruse 模型，自定义光子数
    python main.py --visibility 10 --distance 5 --model kruse --photons 200000
"""

import argparse
import os
import sys

import numpy as np

from src.atmospheric_transport import apply_psf, simulate
from src.visualization import (
    plot_image_comparison,
    plot_psf_2d,
    plot_psf_3d,
    plot_psf_cross_section,
    plot_transmittance_curve,
)


def _load_image(path: str) -> np.ndarray:
    """加载图像文件为 numpy 数组。"""
    from PIL import Image

    img = Image.open(path)
    return np.array(img)


def _save_image(image: np.ndarray, path: str) -> None:
    """将 numpy 数组保存为图像文件。"""
    from PIL import Image

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def main(argv: list[str] | None = None) -> None:
    """命令行主入口。"""
    parser = argparse.ArgumentParser(
        description="大气传输效应模拟工具 - 生成 PSF 并对图像进行卷积处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--visibility", type=float, default=10.0, help="大气能见度 (km)，默认 10"
    )
    parser.add_argument(
        "--distance", type=float, default=5.0, help="传输距离 (km)，默认 5"
    )
    parser.add_argument(
        "--wavelength", type=float, default=10.0, help="工作波长 (μm)，默认 10"
    )
    parser.add_argument(
        "--model",
        choices=["kruse", "kim"],
        default="kim",
        help="消光模型 (kruse/kim)，默认 kim",
    )
    parser.add_argument(
        "--photons", type=int, default=100_000, help="蒙特卡洛光子数，默认 100000"
    )
    parser.add_argument(
        "--asymmetry", type=float, default=0.85, help="不对称因子 g，默认 0.85"
    )
    parser.add_argument(
        "--scattering-ratio",
        type=float,
        default=0.9,
        help="单次散射反照率 ω₀，默认 0.9",
    )
    parser.add_argument(
        "--psf-size", type=int, default=101, help="PSF 矩阵边长，默认 101"
    )
    parser.add_argument(
        "--pixel-scale", type=float, default=0.01, help="像素尺度 (km)，默认 0.01"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--image", type=str, default=None, help="输入图像路径")
    parser.add_argument(
        "--output", type=str, default="output", help="输出目录，默认 output"
    )

    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  大气传输效应模拟 (Atmospheric Transport Simulation)")
    print("=" * 60)
    print(f"  能见度:        {args.visibility} km")
    print(f"  传输距离:      {args.distance} km")
    print(f"  工作波长:      {args.wavelength} μm")
    print(f"  消光模型:      {args.model}")
    print(f"  光子数:        {args.photons}")
    print(f"  不对称因子:    {args.asymmetry}")
    print(f"  散射反照率:    {args.scattering_ratio}")
    print(f"  PSF 矩阵边长: {args.psf_size}")
    print(f"  像素尺度:      {args.pixel_scale} km")
    print("=" * 60)

    # 执行模拟
    print("\n[1/4] 正在执行蒙特卡洛模拟...")
    result = simulate(
        visibility_km=args.visibility,
        distance_km=args.distance,
        wavelength_um=args.wavelength,
        model=args.model,
        n_photons=args.photons,
        asymmetry_factor=args.asymmetry,
        scattering_ratio=args.scattering_ratio,
        psf_size=args.psf_size,
        pixel_scale=args.pixel_scale,
        seed=args.seed,
    )

    print(f"  消光系数: {result['extinction_coeff']:.4f} km⁻¹")
    print(f"  大气透过率: {result['transmittance']:.4f}")

    psf = result["psf"]

    # 生成可视化
    print("\n[2/4] 正在生成 PSF 可视化...")
    plot_psf_2d(psf, save_path=os.path.join(args.output, "psf_2d.png"))
    print(f"  已保存: {args.output}/psf_2d.png")

    plot_psf_cross_section(psf, save_path=os.path.join(args.output, "psf_cross_section.png"))
    print(f"  已保存: {args.output}/psf_cross_section.png")

    plot_psf_3d(psf, save_path=os.path.join(args.output, "psf_3d.png"))
    print(f"  已保存: {args.output}/psf_3d.png")

    print("\n[3/4] 正在生成透过率曲线...")
    plot_transmittance_curve(
        wavelength_um=args.wavelength,
        model=args.model,
        save_path=os.path.join(args.output, "transmittance_curve.png"),
    )
    print(f"  已保存: {args.output}/transmittance_curve.png")

    # 图像卷积
    if args.image:
        print(f"\n[4/4] 正在对图像施加大气传输效应: {args.image}")
        if not os.path.isfile(args.image):
            print(f"  错误: 文件不存在 - {args.image}", file=sys.stderr)
            sys.exit(1)

        original = _load_image(args.image)
        degraded = apply_psf(original, psf)

        output_image_path = os.path.join(args.output, "degraded_image.png")
        _save_image(degraded, output_image_path)
        print(f"  已保存退化图像: {output_image_path}")

        plot_image_comparison(
            original,
            degraded,
            save_path=os.path.join(args.output, "comparison.png"),
        )
        print(f"  已保存对比图: {args.output}/comparison.png")
    else:
        print("\n[4/4] 未指定输入图像，跳过卷积步骤")
        print("  提示: 使用 --image <路径> 指定待处理的图像")

    # 保存 PSF 数据
    psf_data_path = os.path.join(args.output, "psf_data.npy")
    np.save(psf_data_path, psf)
    print(f"\n  已保存 PSF 数据: {psf_data_path}")

    print("\n" + "=" * 60)
    print("  模拟完成！所有结果已保存至:", args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
