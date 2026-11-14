"""Модульные тесты 5-компонентного анализатора ФРТ, слияния окон и планарных метрик качества (nmspect.spectrometry)."""

from __future__ import annotations

import numpy as np
from hepunits import keV, mm

from nmspect.spectrometry import (
    ComponentAnalyzer,
    EnergyWindowConfig,
    PhotonComponent,
    SphericalInsertSpec,
    classify_collimator_passage,
    evaluate_planar_phantom_quality,
    fuse_calibrated_windows,
    get_energy_window_preset,
    infer_emission_bands_from_energies,
)


def test_component_analyzer_five_components_and_tew_additivity() -> None:
    """Проверяет учёт всех 5 компонент PhotonComponent и строгую аддитивность эффективного счёта при TEW."""
    main_win = get_energy_window_preset("peak75_tew_sym", isotope="161Tb")
    energies = np.array(
        [
            74.6 * keV,  # GEOMETRIC в основном окне
            75.0 * keV,  # SEPTAL_PENETRATION в основном окне
            73.0 * keV,  # COLLIMATOR_SCATTER в основном окне
            71.0 * keV,  # PHANTOM_SCATTER в основном окне
            76.0 * keV,  # BACKSCATTER в основном окне
            62.0 * keV,  # PHANTOM_SCATTER в нижнем подокне (перевычитание по компоненте!)
            85.0 * keV,  # BACKSCATTER в верхнем подокне
        ],
        dtype=np.float64,
    )
    components = np.array(
        [
            int(PhotonComponent.GEOMETRIC),
            int(PhotonComponent.SEPTAL_PENETRATION),
            int(PhotonComponent.COLLIMATOR_SCATTER),
            int(PhotonComponent.PHANTOM_SCATTER),
            int(PhotonComponent.BACKSCATTER),
            int(PhotonComponent.PHANTOM_SCATTER),
            int(PhotonComponent.BACKSCATTER),
        ],
        dtype=np.int8,
    )
    offsets = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [8.0, 0.0],
            [15.0, 0.0],
            [35.0, 0.0],
            [16.0, 0.0],
            [38.0, 0.0],
        ],
        dtype=np.float64,
    ) * float(mm)

    analyzer = ComponentAnalyzer(psf_half_extent=60.0 * mm, psf_pixel_size=2.0 * mm, num_radial_bins=30)
    breakdown = analyzer.analyze_window(energies, components, window_config=main_win, psf_offsets=offsets)

    assert breakdown.raw_total_counts == 5.0
    assert breakdown.raw_backscatter_counts == 1.0
    assert breakdown.effective_total_counts > 0.0
    frac_sum = (
        breakdown.geometric_fraction
        + breakdown.septal_fraction
        + breakdown.collimator_scatter_fraction
        + breakdown.phantom_scatter_fraction
        + breakdown.backscatter_fraction
    )
    assert np.isclose(frac_sum, 1.0)

    psf_res = analyzer.compute_psf(energies, offsets, components, window_config=main_win, clip_negative=False)
    reconstructed_total = (
        psf_res.psf_geometric
        + psf_res.psf_septal_penetration
        + psf_res.psf_collimator_scatter
        + psf_res.psf_phantom_scatter
        + psf_res.psf_backscatter
    )
    assert np.allclose(psf_res.psf_total, reconstructed_total)


def test_collimator_passage_window_fusion_and_planar_quality() -> None:
    """Проверяет гексагональную/квадратную апертуру, слияние калиброванных окон и планарные метрики CRC/CNR."""
    birth_pts = np.array([[0.0, 0.0, -200.0], [0.0, 0.0, -200.0]], dtype=np.float64) * float(mm)
    det_pts = np.array([[0.0, 0.0, 20.0], [15.0, 0.0, 20.0]], dtype=np.float64) * float(mm)
    for shape_kind in ("hexagonal", "square"):
        passed = classify_collimator_passage(
            birth_pts,
            det_pts,
            hole_length=35.0 * mm,
            hole_diameter=1.5 * mm,
            septa_thickness=0.2 * mm,
            hole_shape=shape_kind,
        )
        assert list(passed) == [True, False]

    # Слияние двух окон с калибровочными множителями k_1, k_2
    win_a = np.full((10, 10), 20.0, dtype=np.float64)
    win_b = np.full((10, 10), 40.0, dtype=np.float64)
    scat_a = np.full((10, 10), 10.0, dtype=np.float64)
    scat_b = np.full((10, 10), 20.0, dtype=np.float64)
    fused = fuse_calibrated_windows([win_a, win_b], [scat_a, scat_b], calibration_factors=[0.9, 0.75])
    assert np.allclose(fused, (20.0 - 9.0) + (40.0 - 15.0))

    # Планарная оценка сферического очага
    edges_u = np.linspace(-100.0, 100.0, 51, dtype=np.float64) * float(mm)
    edges_v = np.linspace(-100.0, 100.0, 51, dtype=np.float64) * float(mm)
    centers_u = 0.5 * (edges_u[:-1] + edges_u[1:])
    centers_v = 0.5 * (edges_v[:-1] + edges_v[1:])
    grid_u, grid_v = np.meshgrid(centers_u, centers_v, indexing="ij")

    ref_img = np.full((50, 50), 100.0, dtype=np.float64)
    sphere_mask = (grid_u - 20.0 * float(mm)) ** 2 + (grid_v + 10.0 * float(mm)) ** 2 <= (15.0 * float(mm)) ** 2
    ref_img[sphere_mask] = 550.0
    bg_mask = (grid_u < -20.0 * float(mm)) & (np.abs(grid_v) < 40.0 * float(mm))
    # Добавим небольшую вариацию фона для ненулевого СКО
    rng = np.random.default_rng(7)
    noisy_img = ref_img + rng.normal(0.0, 5.0, size=ref_img.shape)

    pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 250.0 * float(mm)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    sphere = SphericalInsertSpec(
        name="S30",
        center_world=(20.0 * float(mm), 0.0, -10.0 * float(mm)),
        diameter=30.0 * float(mm),
        activity_concentration_ratio=10.0,
    )
    qual = evaluate_planar_phantom_quality(
        projection=noisy_img,
        reference_primary=ref_img,
        edges_transverse=edges_u,
        edges_axial=edges_v,
        detector_pose_matrix=pose,
        spheres=[sphere],
        background_mask=bg_mask,
    )
    assert len(qual.sphere_metrics) == 1
    assert qual.sphere_metrics[0].local_contrast > 3.5
    assert qual.sphere_metrics[0].cnr > 40.0
    assert qual.nrmse_vs_primary < 0.05

    # Автоматическая кластеризация линий рождения произвольного изотопа
    sample_emiss = np.concatenate(
        [
            np.full(50, 48.9 * keV),
            np.full(50, 74.6 * keV),
            np.full(20, 290.0 * keV),
        ]
    )
    inferred_bands = infer_emission_bands_from_energies(sample_emiss)
    assert len(inferred_bands) == 3
    covered = sum(bnd.matches(sample_emiss).astype(np.int32) for bnd in inferred_bands)
    assert np.all(covered == 1)

    # Проверка build_component_projections с 5 компонентами
    ev_mat = np.column_stack(
        [
            np.zeros(2),
            np.zeros(2),
            np.array([74.6 * keV, 75.0 * keV]),
            np.array([0.0, 5.0 * mm]),
            np.array([0.0, -5.0 * mm]),
        ]
    )
    stacks = ComponentAnalyzer.build_component_projections(
        events_list=[ev_mat],
        components_list=[np.array([int(PhotonComponent.GEOMETRIC), int(PhotonComponent.BACKSCATTER)], dtype=np.int8)],
        detector_size=(100.0 * mm, 100.0 * mm),
        pixel_size=10.0 * mm,
        energy_windows=[EnergyWindowConfig("w75", 67.1 * keV, 82.0 * keV)],
    )
    assert set(stacks.keys()) == {
        "geometric",
        "septal_penetration",
        "collimator_scatter",
        "phantom_scatter",
        "backscatter",
    }
    assert float(np.sum(stacks["geometric"])) == 1.0
    assert float(np.sum(stacks["backscatter"])) == 1.0
