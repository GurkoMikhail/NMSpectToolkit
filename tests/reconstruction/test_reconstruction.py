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
