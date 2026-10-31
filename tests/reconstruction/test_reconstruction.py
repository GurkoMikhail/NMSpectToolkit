"""Тесты 3D томографической реконструкции (TomographicProjector, FBP, OSEM, MLEM, коррекция затухания)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hepunits import cm, deg, mm

from nmspect.reconstruction import (
    FBPReconstructor,
    MLEMReconstructor,
    OSEMReconstructor,
    TomographicProjector,
    apply_chang_attenuation_correction,
    create_uniform_attenuation_map,
    estimate_attenuation_map_from_projections,
    generate_ordered_subsets,
    get_water_linear_attenuation,
    load_projections,
    save_slices_preview_png,
    save_volume_as_npy,
    save_volume_as_vti,
)


def _create_synthetic_sphere_volume(shape: tuple[int, int, int] = (24, 24, 8)) -> np.ndarray:
    """Создаёт 3D-фантом (nx, nx, nz) с ярким сферическим источником вблизи центра."""
    num_x, num_y, num_z = shape
    grid_x, grid_y, grid_z = np.ogrid[:num_x, :num_y, :num_z]
    center_x, center_y, center_z = num_x / 2.0, num_y / 2.0, num_z / 2.0
    dist_sq = (grid_x - center_x + 1.5) ** 2 + (grid_y - center_y - 2.0) ** 2 + (grid_z - center_z) ** 2
    volume = np.zeros(shape, dtype=np.float64)
    volume[dist_sq <= 3.5**2] = 100.0
    return volume


def test_projector_forward_and_backproject() -> None:
    """Проверяет прямое и обратное проецирование, включая учёт карты затухания и PSF."""
    volume = _create_synthetic_sphere_volume((20, 20, 6))
    angles = np.linspace(0.0, 360.0 * deg, 12, endpoint=False)
    voxel_size = 4.2 * mm

    attenuation_map = create_uniform_attenuation_map(
        shape=volume.shape,
        linear_attenuation=get_water_linear_attenuation("161Tb"),
        voxel_size=voxel_size,
        phantom_radius=30.0 * mm,
    )

    proj_no_atten = TomographicProjector.forward_project(volume, angles, voxel_size=voxel_size)
    proj_with_atten = TomographicProjector.forward_project(
        volume,
        angles,
        attenuation_map=attenuation_map,
        voxel_size=voxel_size,
        psf_fwhm=5.0 * mm,
    )
    assert proj_no_atten.shape == (12, 20, 6)
    assert np.sum(proj_no_atten) > np.sum(proj_with_atten) > 0.0

    backprojected = TomographicProjector.backproject(
        proj_no_atten,
        angles,
        attenuation_map=attenuation_map,
        voxel_size=voxel_size,
    )
    assert backprojected.shape == volume.shape
    assert np.max(backprojected) > 0.0


def test_fbp_osem_and_mlem_reconstruction(tmp_path: Path) -> None:
    """Проверяет восстановление 3D-фантома методами FBP, OSEM и MLEM и сохранение результатов."""
    true_volume = _create_synthetic_sphere_volume((24, 24, 6))
    angles = np.linspace(0.0, 360.0 * deg, 24, endpoint=False)
    voxel_size = 4.2 * mm

    projections = TomographicProjector.forward_project(true_volume, angles, voxel_size=voxel_size)

    # 1. Аналитический метод FBP + коррекция Чанга
    attenuation_map = estimate_attenuation_map_from_projections(
        projections,
        angles=angles,
        linear_attenuation=0.15 / cm,
        voxel_size=voxel_size,
    )
    fbp = FBPReconstructor(
        filter_name="shepp-logan",
        cutoff_frequency=0.9,
        voxel_size=voxel_size,
        attenuation_map=attenuation_map,
        post_smoothing_fwhm=4.5 * mm,
    )
    fbp_res = fbp.reconstruct_with_metadata(projections, angles=angles)
    assert fbp_res.volume.shape == true_volume.shape
    assert fbp_res.attenuation_corrected is True

    corrected_vol, chang_map = apply_chang_attenuation_correction(
        fbp_res.volume,
        attenuation_map=attenuation_map,
        angles=angles,
        voxel_size=voxel_size,
    )
    assert corrected_vol.shape == true_volume.shape
    assert chang_map.shape == true_volume.shape

    # 2. Итерационный метод OSEM с TV-регуляризацией
    subsets = generate_ordered_subsets(24, 4)
    assert len(subsets) == 4

    osem = OSEMReconstructor(
        num_iterations=3,
        num_subsets=4,
        voxel_size=voxel_size,
        attenuation_map=attenuation_map,
        psf_fwhm=4.2 * mm,
        prior_type="tv",
        prior_beta=0.01,
    )
    osem_res = osem.reconstruct_with_metadata(projections, angles=angles)
    assert osem_res.volume.shape == true_volume.shape
    assert np.max(osem_res.volume) > 0.0

    iter_snapshots = osem.reconstruct_iterations(
        projections[:, :, 3],
        angles=angles,
        attenuation_map=attenuation_map[:, :, 3],
        iterations_to_save=[1, 3],
    )
    assert set(iter_snapshots.keys()) == {1, 3}
    assert iter_snapshots[3].volume.shape == (24, 24)

    # Проверка сохранения и повторного чтения
    npy_path = save_volume_as_npy(osem_res.volume, tmp_path / "recon.npy")
    vti_path = save_volume_as_vti(osem_res.volume, tmp_path / "recon.vti", voxel_size=voxel_size)
    png_path = save_slices_preview_png(osem_res.volume, tmp_path / "recon.png")
    assert npy_path.is_file()
    assert vti_path.is_file()
    assert png_path.is_file()

    # Проверка MLEM
    mlem = MLEMReconstructor(num_iterations=2, voxel_size=voxel_size)
    mlem_vol = mlem.reconstruct(projections, angles=angles)
    assert mlem_vol.shape == true_volume.shape

    # Проверка универсального загрузчика проекций из массива
    loaded_proj, loaded_angles, loaded_px = load_projections(projections, angles=angles, pixel_size=voxel_size)
    assert loaded_proj.shape == projections.shape
    assert np.allclose(loaded_angles, angles)
    assert np.isclose(loaded_px, voxel_size)


def test_projector_adjoint_dot_product() -> None:
    """Проверяет строгую эрмитову сопряжённость <A x, y> == <x, A^T y> прямого и обратного проекторов."""
    rng = np.random.default_rng(1042)
    shape_3d = (20, 20, 6)
    num_views = 12
    angles = np.linspace(0.0, 360.0 * deg, num_views, endpoint=False)
    voxel_size = 4.2 * mm

    volume_x = rng.uniform(0.1, 5.0, size=shape_3d)
    proj_y = rng.uniform(0.1, 5.0, size=(num_views, shape_3d[0], shape_3d[2]))
    attenuation_map = create_uniform_attenuation_map(
        shape=shape_3d,
        linear_attenuation=0.15 / cm,
        voxel_size=voxel_size,
        phantom_radius=32.0 * mm,
    )
    axial_gradient = np.linspace(0.7, 1.3, shape_3d[2], dtype=np.float64)[np.newaxis, np.newaxis, :]
    non_uniform_att_3d = attenuation_map * axial_gradient

    test_cases: list[tuple[np.ndarray | None, float | None, bool]] = [
        (None, None, True),
        (None, None, False),
        (attenuation_map, None, True),
        (non_uniform_att_3d, None, True),
        (None, 8.4 * mm, True),
        (attenuation_map, 8.4 * mm, True),
        (non_uniform_att_3d, 8.4 * mm, True),
    ]
    for att_arg, psf_arg, fov_flag in test_cases:
        forward_ax = TomographicProjector.forward_project(
            volume=volume_x,
            angles=angles,
            attenuation_map=att_arg,
            voxel_size=voxel_size,
            psf_fwhm=psf_arg,
            apply_fov_mask=fov_flag,
        )
        adjoint_aty = TomographicProjector.backproject(
            projections=proj_y,
            angles=angles,
            attenuation_map=att_arg,
            voxel_size=voxel_size,
            apply_fov_mask=fov_flag,
            psf_fwhm=psf_arg,
        )
        inner_prod_proj = float(np.sum(forward_ax * proj_y))
        inner_prod_vol = float(np.sum(volume_x * adjoint_aty))
        rel_error = abs(inner_prod_proj - inner_prod_vol) / max(abs(inner_prod_proj), 1e-12)
        assert rel_error < 1e-12

    # Проверка итерационного реконструктора OSEM на неоднородной вдоль Z 3D карте затухания
    osem_non_uniform = OSEMReconstructor(
        num_iterations=2,
        num_subsets=3,
        voxel_size=voxel_size,
        attenuation_map=non_uniform_att_3d,
        psf_fwhm=4.2 * mm,
    )
    recon_non_uniform = osem_non_uniform.reconstruct(proj_y, angles=angles)
    assert recon_non_uniform.shape == shape_3d
    assert float(np.max(recon_non_uniform)) > 0.0


def test_osem_prior_subset_scaling_and_neumann_boundaries() -> None:
    """Проверяет граничные условия Неймана (нулевой суммарный поток) и устойчивость OSEM при сильной TV-регуляризации."""
    rng = np.random.default_rng(2026)
    test_vol = rng.uniform(1.0, 50.0, size=(16, 16, 6))

    for prior_kind in ("quadratic", "tv"):
        recon = OSEMReconstructor(
            num_iterations=3,
            num_subsets=8,
            prior_type=prior_kind,
            prior_beta=0.5,
        )
        grad = recon._compute_prior_gradient_3d(test_vol)
        # Сумма дивергенции потока по всему объёму при однородных условиях Неймана равна нулю
        assert abs(float(np.sum(grad))) < 1e-10

    # Проверка отсутствия паразитной связи между первым (z=0) и последним (z=nz-1) срезами
    slice_vol = np.full((12, 12, 5), 10.0, dtype=np.float64)
    slice_vol[:, :, 0] = 100.0
    recon_tv = OSEMReconstructor(num_iterations=4, num_subsets=8, prior_type="tv", prior_beta=0.5)
    grad_slices = recon_tv._compute_prior_gradient_3d(slice_vol)
    assert np.allclose(grad_slices[:, :, -1], 0.0)

    # Проверка устойчивости OSEM при num_subsets=8 и prior_beta=0.5 (отсутствие взрыва оценки)
    true_vol = _create_synthetic_sphere_volume((24, 24, 6))
    angles = np.linspace(0.0, 360.0 * deg, 24, endpoint=False)
    voxel_size = 4.2 * mm
    projections = TomographicProjector.forward_project(true_vol, angles, voxel_size=voxel_size)

    osem_strong_tv = OSEMReconstructor(
        num_iterations=4,
        num_subsets=8,
        voxel_size=voxel_size,
        prior_type="tv",
        prior_beta=0.5,
    )
    recon_vol = osem_strong_tv.reconstruct(projections, angles=angles)
    true_sum = float(np.sum(true_vol))
    recon_sum = float(np.sum(recon_vol))
    assert 0.7 * true_sum <= recon_sum <= 1.3 * true_sum
    assert float(np.max(recon_vol)) < 2.0 * float(np.max(true_vol))

