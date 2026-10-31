"""Единый параметризованный конвейер постобработки, спектрометрии и экспорта проекционных данных ОФЭКТ."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from hepunits import deg, keV, mm, ns
from numpy.typing import NDArray

from nmspect.detector.projection_builder import generate_projections_from_events
from nmspect.detector.scintillation import ScintillationProcessor
from nmspect.exporters.dicom_nm import DicomNMExporter
from nmspect.exporters.nexus import NeXusExporter
from nmspect.exporters.vtk_paraview import VtkParaviewExporter
from nmspect.io.hdf5_reader import (
    SimulationDataExtractor,
    clear_hdf5_status_flags_if_needed,
    parse_gantry_angle_from_filename,
)
from nmspect.io.legacy_formats import (
    save_projections_as_npy,
    save_projections_as_spr,
    save_spr_as_png,
)
from nmspect.spectrometry.analyzer import SpectrumProcessor
from nmspect.spectrometry.models import EnergyWindowConfig, SpectrumResult
from nmspect.spectrometry.presets import (
    TB161_PEAK_75,
    TC99M_PEAK_140,
    get_energy_window_preset,
    list_available_presets,
    normalize_isotope_name,
)

DEFAULT_RAW_DATA_DIRS: dict[str, str] = {
    "161Tb": "data/raw/output_tb161_dual_cam",
    "99mTc": "data/raw/output_tc99m_dual_cam",
}

DEFAULT_STUDY_PREFIXES: dict[str, str] = {
    "161Tb": "tb161_nema_120views",
    "99mTc": "tc99m_nema_120views",
}

DEFAULT_REFERENCE_ENERGIES: dict[str, float] = {
    "161Tb": TB161_PEAK_75,
    "99mTc": TC99M_PEAK_140,
}


@dataclass(frozen=True)
class CollectedProjectionDataset:
    """Агрегированный набор обработанных сцинтилляционных вспышек по всем угловым ракурсам гантри.

    Attributes:
        sorted_events: Список массивов вспышек `(M_k, 5)` для каждого ракурса, отсортированный по углу.
        sorted_angles: Вектор углов гантри в базовых единицах `hepunits` (радианы).
        all_energies: Сводный массив энергий всех зарегистрированных вспышек по всем ракурсам.
        all_scatter_orders: Сводный массив кратностей комптоновского рассеяния по всем ракурсам.
        detector_size: Габариты кристалла детектора `(size_x, size_y, thickness)` в базовых единицах `hepunits`.
        simulation_metadata: Словарь паспортных метаданных симуляции из HDF5.
    """

    sorted_events: list[NDArray[np.float64]]
    sorted_angles: NDArray[np.float64]
    all_energies: NDArray[np.float64]
    all_scatter_orders: NDArray[np.int32]
    detector_size: tuple[float, float, float]
    simulation_metadata: dict[str, Any]


def resolve_data_directory(
    data_dir: str | Path | None = None,
    isotope: str = "161Tb",
) -> Path:
    """Определяет и проверяет путь к директории с входными HDF5-файлами симуляции.

    Args:
        data_dir: Явно указанный путь к директории или None для выбора стандартного пути по изотопу.
        isotope: Обозначение радионуклида ('161Tb' или '99mTc').

    Returns:
        Проверенный путь `Path` к существующей директории.
    """
    if data_dir is not None:
        candidate = Path(data_dir)
        if not candidate.is_dir():
            raise FileNotFoundError(f"Указанная директория входных данных не найдена: {candidate}")
        return candidate

    canonical_isotope = normalize_isotope_name(isotope)
    default_rel_path = DEFAULT_RAW_DATA_DIRS[canonical_isotope]
    candidate = Path(default_rel_path)
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Директория сырых данных для {canonical_isotope} не найдена по пути '{candidate}'. "
        "Укажите путь явно через параметр --data-dir."
    )


def resolve_orbit_radius(
    explicit_orbit_radius: float | None,
    simulation_metadata: dict[str, Any],
    default_orbit_radius: float = 200.6 * mm,
) -> float:
    """Определяет радиус орбиты вращения детекторов (в базовых единицах hepunits).

    Приоритет источников:
    1. Явно переданный `explicit_orbit_radius`;
    2. Поле `orbit_radius` или матрица трансформации детектора в `simulation_metadata['gantry']`;
    3. Резервное значение `default_orbit_radius`.

    Args:
        explicit_orbit_radius: Явно заданный радиус орбиты (в базовых единицах hepunits) или None.
        simulation_metadata: Словарь метаданных симуляции.
        default_orbit_radius: Значение по умолчанию в единицах hepunits.

    Returns:
        Радиус орбиты детектора в базовых единицах hepunits (мм).
    """
    if explicit_orbit_radius is not None:
        if explicit_orbit_radius <= 0.0:
            raise ValueError(f"Радиус орбиты должен быть положительным, получено: {explicit_orbit_radius}")
        return float(explicit_orbit_radius)

    gantry_meta = simulation_metadata.get("gantry")
    if isinstance(gantry_meta, dict):
        if "orbit_radius" in gantry_meta:
            raw_radius = float(gantry_meta["orbit_radius"])
            if raw_radius > 0.0:
                return raw_radius * mm
        pose_mat = gantry_meta.get("pose_matrix_detector_1")
        if pose_mat is not None:
            pose_arr = np.asarray(pose_mat, dtype=np.float64)
            if pose_arr.shape == (4, 4):
                radial_dist = float(np.hypot(pose_arr[0, 3], pose_arr[1, 3]))
                if radial_dist > 0.0:
                    return radial_dist * mm

    return float(default_orbit_radius)


def collect_projection_data(
    data_dir: str | Path,
    time_resolution: float = 230.0 * ns,
    energy_resolution: float = 0.095,
    intrinsic_spatial_resolution: float = 3.8 * mm,
    reference_energy: float | None = None,
    rng_seed: int | None = None,
) -> CollectedProjectionDataset:
    """Читает все HDF5-файлы ракурсов из директории и формирует сцинтилляционные вспышки.

    Args:
        data_dir: Путь к директории с `.hdf`/`.h5` файлами угловых позиций гантри.
        time_resolution: Временное окно разрешения сцинтиллятора (в базовых единицах hepunits).
        energy_resolution: Относительное энергетическое разрешение FWHM (доля, например 0.095 для 9.5%).
        intrinsic_spatial_resolution: Внутреннее пространственное разрешение FWHM кристалла (в hepunits).
        reference_energy: Опорная энергия для масштабирования энергетического разрешения (в hepunits).
        rng_seed: Зерно генератора случайных чисел для воспроизводимости.

    Returns:
        Объект `CollectedProjectionDataset` с отсортированными по углам событиями и метаданными.
    """
    resolved_dir = Path(data_dir)
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Директория данных не найдена: {resolved_dir}")

    hdf_files = sorted(
        list(resolved_dir.glob("*.hdf")) + list(resolved_dir.glob("*.h5")) + list(resolved_dir.glob("*.hdf5"))
    )
    if not hdf_files:
        raise FileNotFoundError(f"В директории {resolved_dir} не найдено ни одного HDF5-файла (*.hdf, *.h5).")

    scint_processor = ScintillationProcessor(
        time_resolution=time_resolution,
        energy_resolution=energy_resolution,
        intrinsic_spatial_resolution=intrinsic_spatial_resolution,
        reference_energy=reference_energy,
        rng=rng_seed,
    )

    angle_event_pairs: list[tuple[float, NDArray[np.float64]]] = []
    all_energies_chunks: list[NDArray[np.float64]] = []
    all_orders_chunks: list[NDArray[np.int32]] = []
    detector_size: tuple[float, float, float] | None = None
    simulation_metadata: dict[str, Any] = {}

    for hdf_path in hdf_files:
        clear_hdf5_status_flags_if_needed(hdf_path)
        extractor = SimulationDataExtractor(hdf_path)

        if not simulation_metadata:
            simulation_metadata = extractor.get_simulation_metadata()
        if detector_size is None:
            detector_size = extractor.get_detector_size()

        detector_names = extractor.get_detectors_names()
        detector_angles = extractor.get_detectors_angles()
        parsed_base_angle = parse_gantry_angle_from_filename(hdf_path.name)

        for det_idx, det_name in enumerate(detector_names):
            if det_idx < len(detector_angles):
                view_angle = float(detector_angles[det_idx])
            elif parsed_base_angle is not None:
                view_angle = float(parsed_base_angle + det_idx * 180.0 * deg)
            else:
                view_angle = float(det_idx * 180.0 * deg)

            normalized_angle = float(view_angle % (360.0 * deg))
            raw_acts = extractor.extract_data(det_name)
            raw_orders = extractor.get_scattering_orders(det_name)

            processed_events, processed_orders = scint_processor.process_detector_acts(
                raw_acts,
                scattering_orders=raw_orders,
            )
            angle_event_pairs.append((normalized_angle, processed_events))
            if processed_events.shape[0] > 0:
                all_energies_chunks.append(processed_events[:, 2])
                all_orders_chunks.append(processed_orders)

    angle_event_pairs.sort(key=lambda pair: pair[0])
    sorted_angles = np.array([pair[0] for pair in angle_event_pairs], dtype=np.float64)
    sorted_events = [pair[1] for pair in angle_event_pairs]

    if all_energies_chunks:
        all_energies = np.concatenate(all_energies_chunks).astype(np.float64)
        all_scatter_orders = np.concatenate(all_orders_chunks).astype(np.int32)
    else:
        all_energies = np.empty((0,), dtype=np.float64)
        all_scatter_orders = np.empty((0,), dtype=np.int32)

    final_detector_size = detector_size if detector_size is not None else (540.0 * mm, 400.0 * mm, 9.5 * mm)

    return CollectedProjectionDataset(
        sorted_events=sorted_events,
        sorted_angles=sorted_angles,
        all_energies=all_energies,
        all_scatter_orders=all_scatter_orders,
        detector_size=final_detector_size,
        simulation_metadata=simulation_metadata,
    )


def export_projection_study(
    study_name: str,
    projections: NDArray[np.floating],
    angles: Sequence[float] | NDArray[np.floating],
    spectrum_result: SpectrumResult,
    pixel_size: float,
    detector_size: Sequence[float],
    orbit_radius: float,
    window_config: EnergyWindowConfig | None,
    simulation_metadata: dict[str, Any],
    output_root: str | Path = "results",
    per_window_projections: Mapping[str, NDArray[np.floating]] | None = None,
    export_legacy_formats: bool = True,
) -> dict[str, Path]:
    """Экспортирует обработанное проекционное исследование во все целевые форматы в каталог `results/`.

    Args:
        study_name: Базовое имя исследования (например, 'tb161_nema_120views_peak75_tew').
        projections: Трёхмерный массив проекций `(num_views, height, width)`.
        angles: Массив углов гантри в базовых единицах `hepunits` (радианы).
        spectrum_result: Результат трёхуровневого спектрометрического анализа.
        pixel_size: Размер пикселя проекции в базовых единицах `hepunits`.
        detector_size: Габариты детектора `(size_x, size_y, thickness)` в `hepunits`.
        orbit_radius: Радиус орбиты вращения детекторов в `hepunits`.
        window_config: Конфигурация энергетического окна.
        simulation_metadata: Паспортные метаданные симуляции.
        output_root: Корневая директория результатов (по умолчанию `'results'`).
        per_window_projections: Опциональный словарь проекций по отдельным энергетическим подокнам.
        export_legacy_formats: Сохранять ли также массивы `.npy`, `.spr` и `.png` в `results/{numpy,spr,png}`.

    Returns:
        Словарь путей к созданным файлам (`{'nexus': ..., 'dicom': ..., 'paraview': ..., ...}`).
    """
    root_dir = Path(output_root)
    nexus_dir = root_dir / "nexus"
    dicom_dir = root_dir / "dicom"
    paraview_dir = root_dir / "paraview"
    nexus_dir.mkdir(parents=True, exist_ok=True)
    dicom_dir.mkdir(parents=True, exist_ok=True)
    paraview_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, Path] = {}

    nexus_path = nexus_dir / f"{study_name}.nxs"
    nexus_exporter = NeXusExporter(nexus_path)
    exported_paths["nexus"] = nexus_exporter.export(
        projections=projections,
        angles=angles,
        spectrum_result=spectrum_result,
        pixel_size=pixel_size,
        detector_size=detector_size,
        window_config=window_config,
        metadata=simulation_metadata,
        per_window_projections=per_window_projections,
    )

    dicom_path = dicom_dir / f"{study_name}.dcm"
    dicom_exporter = DicomNMExporter(dicom_path)
    exported_paths["dicom"] = dicom_exporter.export(
        projections=projections,
        angles=angles,
        pixel_size=pixel_size,
        window_config=window_config,
        orbit_radius=orbit_radius,
        metadata=simulation_metadata,
    )

    vtm_path = paraview_dir / f"{study_name}.vtm"
    vtk_exporter = VtkParaviewExporter(vtm_path)
    exported_paths["paraview"] = vtk_exporter.export(
        projections=projections,
        angles=angles,
        pixel_size=pixel_size,
        detector_size=detector_size,
        orbit_radius=orbit_radius,
        window_config=window_config,
        metadata=simulation_metadata,
    )

    if export_legacy_formats:
        numpy_dir = root_dir / "numpy"
        spr_dir = root_dir / "spr" / study_name
        png_dir = root_dir / "png" / study_name
        numpy_dir.mkdir(parents=True, exist_ok=True)

        exported_paths["numpy_projections"] = save_projections_as_npy(
            projections,
            numpy_dir / f"{study_name}.npy",
        )
        converted_angles = np.asarray(angles, dtype=np.float64) / deg
        angles_npy_path = numpy_dir / f"{study_name}_angles.npy"
        np.save(angles_npy_path, converted_angles)
        exported_paths["numpy_angles"] = angles_npy_path

        exported_paths["spr_dir"] = save_projections_as_spr(projections, spr_dir)
        proj_height, proj_width = int(projections.shape[1]), int(projections.shape[2])
        save_spr_as_png(spr_dir, png_dir, image_shape=(proj_height, proj_width))
        exported_paths["png_dir"] = png_dir

    return exported_paths


def run_isotope_nema_pipeline(
    isotope: str = "161Tb",
    data_dir: str | Path | None = None,
    preset_keys: Sequence[str] | None = None,
    pixel_size: float = 4.2 * mm,
    orbit_radius: float | None = None,
    time_resolution: float = 230.0 * ns,
    energy_resolution: float = 0.095,
    intrinsic_spatial_resolution: float = 3.8 * mm,
    output_root: str | Path = "results",
    study_prefix: str | None = None,
    export_legacy_formats: bool = True,
    rng_seed: int | None = None,
) -> dict[str, dict[str, Path]]:
    """Выполняет полный конвейер обработки сырых данных симуляции ОФЭКТ для выбранного радионуклида.

    Args:
        isotope: Обозначение радионуклида ('161Tb' или '99mTc').
        data_dir: Директория входных HDF5-файлов (если None — выбирается автоматически из `data/raw/`).
        preset_keys: Список ключей пресетов энергетических окон (если None — обрабатываются все пресеты изотопа).
        pixel_size: Размер пикселя проекционной матрицы (в базовых единицах hepunits).
        orbit_radius: Явный радиус орбиты детекторов (в базовых единицах hepunits) или None для чтения из метаданных.
        time_resolution: Временное разрешение детектора (в базовых единицах hepunits).
        energy_resolution: Относительное энергетическое разрешение FWHM.
        intrinsic_spatial_resolution: Внутреннее пространственное разрешение FWHM кристалла (в hepunits).
        output_root: Корневая директория сохранения результатов (`results/`).
        study_prefix: Префикс имени выходных файлов.
        export_legacy_formats: Сохранять ли дополнительные файлы `.npy`, `.spr` и `.png`.
        rng_seed: Зерно генератора случайных чисел.

    Returns:
        Словарь `{preset_key: {'nexus': Path, 'dicom': Path, 'paraview': Path, ...}}`.
    """
    canonical_isotope = normalize_isotope_name(isotope)
    resolved_data_dir = resolve_data_directory(data_dir, canonical_isotope)
    ref_energy = DEFAULT_REFERENCE_ENERGIES[canonical_isotope]
    prefix = study_prefix if study_prefix is not None else DEFAULT_STUDY_PREFIXES[canonical_isotope]

    selected_presets = (
        list(preset_keys) if preset_keys is not None and len(preset_keys) > 0 else list_available_presets(canonical_isotope)
    )

    dataset = collect_projection_data(
        data_dir=resolved_data_dir,
        time_resolution=time_resolution,
        energy_resolution=energy_resolution,
        intrinsic_spatial_resolution=intrinsic_spatial_resolution,
        reference_energy=ref_energy,
        rng_seed=rng_seed,
    )

    effective_orbit_radius = resolve_orbit_radius(
        explicit_orbit_radius=orbit_radius,
        simulation_metadata=dataset.simulation_metadata,
    )

    max_energy = 250.0 * keV if canonical_isotope == "99mTc" else 300.0 * keV
    spectrum_processor = SpectrumProcessor(energy_range=(0.0 * keV, max_energy), num_bins=1024)

    results_by_preset: dict[str, dict[str, Path]] = {}
    for preset_key in selected_presets:
        windows = get_energy_window_preset(preset_key, isotope=canonical_isotope)
        combined_projections, per_window_projections = generate_projections_from_events(
            events_list=dataset.sorted_events,
            detector_size=dataset.detector_size,
            pixel_size=pixel_size,
            energy_windows=windows,
            clip_negative=True,
        )

        primary_window = windows[0]
        spectrum_result = spectrum_processor.analyze(
            energies=dataset.all_energies,
            scatter_orders=dataset.all_scatter_orders,
            window_config=primary_window,
        )

        study_name = f"{prefix}_{preset_key}"
        exported = export_projection_study(
            study_name=study_name,
            projections=combined_projections,
            angles=dataset.sorted_angles,
            spectrum_result=spectrum_result,
            pixel_size=pixel_size,
            detector_size=dataset.detector_size,
            orbit_radius=effective_orbit_radius,
            window_config=primary_window,
            simulation_metadata=dataset.simulation_metadata,
            output_root=output_root,
            per_window_projections=per_window_projections,
            export_legacy_formats=export_legacy_formats,
        )
        results_by_preset[preset_key] = exported

    return results_by_preset
