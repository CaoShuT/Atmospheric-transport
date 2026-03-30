"""大气传输效应模拟核心模块的单元测试。"""

import numpy as np
import pytest

from src.atmospheric_transport import (
    apply_psf,
    atmospheric_transmittance,
    extinction_coefficient,
    monte_carlo_psf,
    simulate,
)


# ────────────────────────────────────────────────────────────────────
# 消光系数测试
# ────────────────────────────────────────────────────────────────────


class TestExtinctionCoefficient:
    """消光系数计算测试。"""

    def test_positive_visibility(self):
        beta = extinction_coefficient(10.0, 10.0, "kim")
        assert beta > 0

    def test_higher_visibility_lower_extinction(self):
        beta_low_vis = extinction_coefficient(1.0, 10.0, "kim")
        beta_high_vis = extinction_coefficient(20.0, 10.0, "kim")
        assert beta_low_vis > beta_high_vis

    def test_kruse_model(self):
        beta = extinction_coefficient(10.0, 10.0, "kruse")
        assert beta > 0

    def test_kim_model(self):
        beta = extinction_coefficient(10.0, 10.0, "kim")
        assert beta > 0

    def test_invalid_visibility(self):
        with pytest.raises(ValueError, match="能见度必须大于 0"):
            extinction_coefficient(0, 10.0)

    def test_negative_visibility(self):
        with pytest.raises(ValueError, match="能见度必须大于 0"):
            extinction_coefficient(-1.0, 10.0)

    def test_invalid_model(self):
        with pytest.raises(ValueError, match="不支持的模型"):
            extinction_coefficient(10.0, 10.0, "invalid")

    def test_kim_very_low_visibility(self):
        """Kim 模型低能见度 (< 0.5 km) 时 q=0。"""
        beta = extinction_coefficient(0.3, 10.0, "kim")
        assert beta > 0

    def test_kim_medium_visibility(self):
        """Kim 模型中等能见度 (1-6 km)。"""
        beta = extinction_coefficient(3.0, 10.0, "kim")
        assert beta > 0

    def test_kruse_high_visibility(self):
        """Kruse 模型高能见度 (> 50 km)。"""
        beta = extinction_coefficient(60.0, 10.0, "kruse")
        assert beta > 0


# ────────────────────────────────────────────────────────────────────
# 大气透过率测试
# ────────────────────────────────────────────────────────────────────


class TestAtmosphericTransmittance:
    """大气透过率计算测试。"""

    def test_zero_distance(self):
        tau = atmospheric_transmittance(10.0, 0.0)
        assert tau == 1.0

    def test_positive_distance(self):
        tau = atmospheric_transmittance(10.0, 5.0)
        assert 0 < tau < 1

    def test_transmittance_decreases_with_distance(self):
        tau_near = atmospheric_transmittance(10.0, 1.0)
        tau_far = atmospheric_transmittance(10.0, 10.0)
        assert tau_near > tau_far

    def test_transmittance_increases_with_visibility(self):
        tau_low_vis = atmospheric_transmittance(1.0, 5.0)
        tau_high_vis = atmospheric_transmittance(20.0, 5.0)
        assert tau_high_vis > tau_low_vis

    def test_negative_distance_error(self):
        with pytest.raises(ValueError, match="传输距离不能为负数"):
            atmospheric_transmittance(10.0, -1.0)

    def test_transmittance_range(self):
        """透过率应在 [0, 1] 范围内。"""
        for vis in [0.5, 1.0, 5.0, 10.0, 50.0]:
            for dist in [0.1, 1.0, 5.0, 20.0]:
                tau = atmospheric_transmittance(vis, dist)
                assert 0 <= tau <= 1


# ────────────────────────────────────────────────────────────────────
# 蒙特卡洛 PSF 测试
# ────────────────────────────────────────────────────────────────────


class TestMonteCarloPSF:
    """蒙特卡洛 PSF 生成测试。"""

    def test_psf_shape(self):
        psf = monte_carlo_psf(n_photons=1000, psf_size=51, seed=42)
        assert psf.shape == (51, 51)

    def test_psf_normalized(self):
        psf = monte_carlo_psf(n_photons=5000, psf_size=51, seed=42)
        assert abs(psf.sum() - 1.0) < 0.01

    def test_psf_non_negative(self):
        psf = monte_carlo_psf(n_photons=1000, psf_size=51, seed=42)
        assert np.all(psf >= 0)

    def test_psf_center_peak(self):
        """PSF 中心应有最大值（直射光子）。"""
        psf = monte_carlo_psf(
            n_photons=10000,
            extinction_coeff=0.5,
            scattering_ratio=0.5,
            psf_size=51,
            seed=42,
        )
        center = 51 // 2
        assert psf[center, center] == psf.max()

    def test_psf_even_size_adjusted(self):
        """偶数尺寸应自动调整为奇数。"""
        psf = monte_carlo_psf(n_photons=1000, psf_size=50, seed=42)
        assert psf.shape[0] % 2 == 1

    def test_psf_reproducible(self):
        """相同种子应产生相同结果。"""
        psf1 = monte_carlo_psf(n_photons=1000, psf_size=31, seed=123)
        psf2 = monte_carlo_psf(n_photons=1000, psf_size=31, seed=123)
        np.testing.assert_array_equal(psf1, psf2)


# ────────────────────────────────────────────────────────────────────
# 图像卷积测试
# ────────────────────────────────────────────────────────────────────


class TestApplyPSF:
    """图像卷积测试。"""

    def test_grayscale_image(self):
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        psf = np.zeros((11, 11))
        psf[5, 5] = 1.0  # Delta PSF
        result = apply_psf(image, psf)
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_color_image(self):
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        psf = np.zeros((11, 11))
        psf[5, 5] = 1.0
        result = apply_psf(image, psf)
        assert result.shape == image.shape

    def test_delta_psf_identity(self):
        """Delta PSF 不应改变图像。"""
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        psf = np.zeros((11, 11))
        psf[5, 5] = 1.0
        result = apply_psf(image, psf)
        np.testing.assert_array_almost_equal(
            result.astype(float), image.astype(float), decimal=0
        )

    def test_invalid_dimensions(self):
        image = np.zeros((10,))
        psf = np.zeros((3, 3))
        with pytest.raises(ValueError, match="不支持的图像维度"):
            apply_psf(image, psf)

    def test_float_image(self):
        """浮点类型图像应正常处理。"""
        image = np.random.rand(32, 32).astype(np.float32)
        psf = np.zeros((11, 11))
        psf[5, 5] = 1.0
        result = apply_psf(image, psf)
        assert result.dtype == np.float32


# ────────────────────────────────────────────────────────────────────
# simulate 集成测试
# ────────────────────────────────────────────────────────────────────


class TestSimulate:
    """模拟管线集成测试。"""

    def test_simulate_returns_dict(self):
        result = simulate(n_photons=1000, psf_size=21, seed=42)
        assert "extinction_coeff" in result
        assert "transmittance" in result
        assert "psf" in result
        assert "params" in result

    def test_simulate_psf_shape(self):
        result = simulate(n_photons=1000, psf_size=31, seed=42)
        assert result["psf"].shape == (31, 31)

    def test_simulate_transmittance_range(self):
        result = simulate(n_photons=1000, psf_size=21, seed=42)
        assert 0 < result["transmittance"] <= 1

    def test_simulate_extinction_positive(self):
        result = simulate(n_photons=1000, psf_size=21, seed=42)
        assert result["extinction_coeff"] > 0
