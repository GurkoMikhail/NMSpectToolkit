"""Экспорт результатов ОФЭКТ (проекций, трёхуровневых спектров и метаданных) в стандарт HDF5 NeXus (.nxs)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from hepunits import keV, mm
from numpy.typing import NDArray

from nmspect.spectrometry.models import EnergyWindowConfig, SpectrumResult


class NeXusExporter:
    """Экспортирует стек проекций, спектры, 6 метрик качества и метаданные симуляции в контейнер NeXus HDF5."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        """Инициализирует экспортер NeXus.

        Args:
            output_path: Опциональный путь к целевому файлу `.nxs` (может быть также передан в `export`).
        """
        self.output_path: Path | None = Path(output_path) if output_path is not None else None

    @staticmethod
    def _write_metadata_group(target_group: h5py.Group, metadata_dict: dict[str, Any]) -> None:
        """Рекурсивно записывает словарь метаданных симуляции в группу HDF5."""
        for meta_key, meta_val in metadata_dict.items():
            if meta_val is None:
                continue
            if isinstance(meta_val, dict):
                sub_group = target_group.create_group(str(meta_key))
                sub_group.attrs["NX_class"] = "NXcollection"
                NeXusExporter._write_metadata_group(sub_group, meta_val)
            elif isinstance(meta_val, (str, bytes)):
                str_val = meta_val.decode("utf-8", errors="replace") if isinstance(meta_val, bytes) else meta_val
                target_group.create_dataset(str(meta_key), data=str_val)
            elif isinstance(meta_val, (int, float, bool, np.number)):
                target_group.create_dataset(str(meta_key), data=meta_val)
            elif isinstance(meta_val, (list, tuple, np.ndarray)):
                arr_val = np.asarray(meta_val)
                if arr_val.dtype.kind in ("U", "S", "O"):
                    target_group.attrs[str(meta_key)] = str(meta_val)
                else:
                    target_group.create_dataset(str(meta_key), data=arr_val)

    def export(
        self,
        projections: NDArray[np.floating],
        angles: Sequence[float] | NDArray[np.floating],
        spectrum_result: SpectrumResult | None = None,
        pixel_size: float = 4.2 * mm,
        detector_size: Sequence[float] | None = None,
        window_config: EnergyWindowConfig | None = None,
        metadata: dict[str, Any] | None = None,
        per_window_projections: Mapping[str, NDArray[np.floating]] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Записывает данные томографического исследования ОФЭКТ в файл стандарта NeXus (`.nxs`).

        Args:
            projections: 3D стек проекций формы `(views, nx, ny)` или 2D срез `(nx, ny)`.
            angles: Вектор углов поворота детектора в базовых единицах hepunits (радианах).
            spectrum_result: Результат трёхуровневого спектрометрического анализа `SpectrumResult`.
            pixel_size: Размер пикселя проекции в базовых единицах hepunits (mm).
            detector_size: Габариты кристалла детектора `(size_x, size_y, size_z)` в hepunits (mm).
            window_config: Конфигурация активного энергетического окна `EnergyWindowConfig`.
            metadata: Словарь паспортных метаданных симуляции и установки.
            per_window_projections: Опциональный словарь 3D-стеков проекций по отдельным окнам.
            output_path: Целевой путь сохранения файла `.nxs` (переопределяет путь из конструктора).

        Returns:
            Полный путь к созданному файлу `.nxs`.
        """
        target_path = Path(output_path) if output_path is not None else self.output_path
        if target_path is None:
            raise ValueError("Не указан путь сохранения файла NeXus (output_path).")
        if pixel_size <= 0.0:
            raise ValueError(f"Размер пикселя должен быть строго положительным: {pixel_size}")

        proj_arr = np.asarray(projections, dtype=np.float32)
        if proj_arr.ndim == 2:
            proj_arr = proj_arr[np.newaxis, ...]
        elif proj_arr.ndim != 3 or proj_arr.size == 0:
            raise ValueError(f"Ожидался непустой 2D или 3D массив проекций, получена форма: {proj_arr.shape}")

        angles_arr = np.asarray(angles, dtype=np.float64).ravel()
        if angles_arr.size != proj_arr.shape[0]:
            raise ValueError(
                f"Число углов ({angles_arr.size}) не совпадает с числом проекций ({proj_arr.shape[0]})."
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        meta_dict = metadata if metadata is not None else {}
        active_window = (
            window_config
            if window_config is not None
            else (spectrum_result.window_config if spectrum_result is not None else None)
        )

        with h5py.File(target_path, "w") as h5_file:
            h5_file.attrs["default"] = "entry"
            h5_file.attrs["file_name"] = target_path.name
            h5_file.attrs["file_time"] = str(np.datetime64("now"))
            h5_file.attrs["HDF5_Version"] = h5py.version.hdf5_version
            h5_file.attrs["NeXus_version"] = "4.3.0"

            entry = h5_file.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            entry.attrs["default"] = "data"

            title_str = str(meta_dict.get("title", "SPECT Tomographic Simulation"))
            if active_window is not None and active_window.primary_peak is not None:
                peak_val = float(active_window.primary_peak / keV)
                title_str = f"{title_str} [{active_window.name}, peak {peak_val:.1f} keV]"
            entry.create_dataset("title", data=title_str)
            entry.create_dataset("definition", data="NXtomo")

            # 1. Описание инструмента (NXinstrument)
            instrument = entry.create_group("instrument")
            instrument.attrs["NX_class"] = "NXinstrument"
            instrument.create_dataset("name", data=str(meta_dict.get("scanner_name", "GE Discovery NM/CT 670")))

            source_grp = instrument.create_group("source")
            source_grp.attrs["NX_class"] = "NXsource"
            source_meta = meta_dict.get("source", {})
            radionuclide_name = (
                source_meta.get("radionuclide", meta_dict.get("radionuclide", "161Tb"))
                if isinstance(source_meta, dict)
                else meta_dict.get("radionuclide", "161Tb")
            )
            source_grp.create_dataset("name", data=str(radionuclide_name))
            source_grp.create_dataset("type", data="Radioactive Isotope")

            collimator_grp = instrument.create_group("collimator")
            collimator_grp.attrs["NX_class"] = "NXcollimator"
            collimator_grp.create_dataset(
                "type", data=str(meta_dict.get("collimator_type", "Parallel Hole (LEHR)"))
            )

            detector_grp = instrument.create_group("detector")
            detector_grp.attrs["NX_class"] = "NXdetector"
            detector_grp.create_dataset(
                "scintillator_material", data=str(meta_dict.get("crystal_material", "NaI(Tl)"))
            )

            ds_pixel = detector_grp.create_dataset("pixel_size", data=float(pixel_size / mm))
            ds_pixel.attrs["units"] = "mm"

            detector_grp.create_dataset("matrix_size", data=np.array(proj_arr.shape[1:], dtype=np.int32))

            if detector_size is not None:
                det_dims = np.asarray([float(val / mm) for val in detector_size], dtype=np.float64)
                ds_det_size = detector_grp.create_dataset("detector_size", data=det_dims)
                ds_det_size.attrs["units"] = "mm"

            if active_window is not None:
                win_grp = instrument.create_group("energy_window")
                win_grp.attrs["NX_class"] = "NXcollection"
                win_grp.create_dataset("name", data=active_window.name)
                win_grp.create_dataset("description", data=active_window.description)

                ds_low = win_grp.create_dataset("lower_bound", data=float(active_window.lower_bound))
                ds_low.attrs["units"] = "hepunits"

                ds_high = win_grp.create_dataset("upper_bound", data=float(active_window.upper_bound))
                ds_high.attrs["units"] = "hepunits"

                if active_window.primary_peak is not None:
                    ds_peak = win_grp.create_dataset("primary_peak", data=float(active_window.primary_peak))
                    ds_peak.attrs["units"] = "hepunits"

                win_grp.create_dataset("is_scatter_window", data=bool(active_window.is_scatter_window))
                win_grp.create_dataset("scatter_weight", data=float(active_window.scatter_weight))

                if active_window.associated_windows:
                    assoc_grp = win_grp.create_group("associated_windows")
                    assoc_grp.attrs["NX_class"] = "NXcollection"
                    for assoc_win in active_window.associated_windows:
                        sub_win = assoc_grp.create_group(assoc_win.name)
                        sub_win.attrs["NX_class"] = "NXcollection"
                        ds_sub_low = sub_win.create_dataset(
                            "lower_bound", data=float(assoc_win.lower_bound)
                        )
                        ds_sub_low.attrs["units"] = "hepunits"
                        ds_sub_high = sub_win.create_dataset(
                            "upper_bound", data=float(assoc_win.upper_bound)
                        )
                        ds_sub_high.attrs["units"] = "hepunits"
                        sub_win.create_dataset("scatter_weight", data=float(assoc_win.scatter_weight))

            # 2. Группа проекционных данных (NXdata)
            data_grp = entry.create_group("data")
            data_grp.attrs["NX_class"] = "NXdata"
            data_grp.attrs["signal"] = "data"
            data_grp.attrs["axes"] = ["gantry_angle", "x", "y"]

            ds_proj = data_grp.create_dataset(
                "data",
                data=proj_arr,
                compression="gzip",
                compression_opts=4,
            )
            ds_proj.attrs["units"] = "counts"
            ds_proj.attrs["total_counts"] = float(np.sum(proj_arr))

            ds_rot_angle = data_grp.create_dataset("rotation_angle", data=angles_arr)
            ds_rot_angle.attrs["units"] = "hepunits"

            ds_gantry_angle = data_grp.create_dataset("gantry_angle", data=angles_arr)
            ds_gantry_angle.attrs["units"] = "hepunits"

            num_x, num_y = proj_arr.shape[1], proj_arr.shape[2]
            step_val = float(pixel_size / mm)
            coords_x = (np.arange(num_x, dtype=np.float32) - 0.5 * float(num_x - 1)) * step_val
            coords_y = (np.arange(num_y, dtype=np.float32) - 0.5 * float(num_y - 1)) * step_val

            ds_coord_x = data_grp.create_dataset("x", data=coords_x)
            ds_coord_x.attrs["units"] = "mm"
            ds_coord_y = data_grp.create_dataset("y", data=coords_y)
            ds_coord_y.attrs["units"] = "mm"

            entry["data_projections"] = h5py.SoftLink("/entry/data")

            if per_window_projections:
                per_win_grp = entry.create_group("windows_projections")
                per_win_grp.attrs["NX_class"] = "NXcollection"
                for win_name, win_stack in per_window_projections.items():
                    ds_win_proj = per_win_grp.create_dataset(
                        str(win_name),
                        data=np.asarray(win_stack, dtype=np.float32),
                        compression="gzip",
                        compression_opts=4,
                    )
                    ds_win_proj.attrs["units"] = "counts"

            # 3. Группа спектрометрического анализа и 6 метрик качества (NXdata)
            if spectrum_result is not None:
                spec_grp = entry.create_group("spectrum")
                spec_grp.attrs["NX_class"] = "NXdata"
                spec_grp.attrs["signal"] = "total"
                spec_grp.attrs["axes"] = ["energy"]

                ds_energy = spec_grp.create_dataset(
                    "energy",
                    data=np.asarray(spectrum_result.energy_bins, dtype=np.float64),
                )
                ds_energy.attrs["units"] = "hepunits"

                ds_edges = spec_grp.create_dataset(
                    "bin_edges",
                    data=np.asarray(spectrum_result.bin_edges, dtype=np.float64),
                )
                ds_edges.attrs["units"] = "hepunits"

                ds_total = spec_grp.create_dataset(
                    "total",
                    data=np.asarray(spectrum_result.total_spectrum, dtype=np.float64),
                    compression="gzip",
                )
                ds_total.attrs["units"] = "counts"

                ds_primary = spec_grp.create_dataset(
                    "primary",
                    data=np.asarray(spectrum_result.primary_spectrum, dtype=np.float64),
                    compression="gzip",
                )
                ds_primary.attrs["units"] = "counts"

                ds_scatter = spec_grp.create_dataset(
                    "scatter",
                    data=np.asarray(spectrum_result.scatter_spectrum, dtype=np.float64),
                    compression="gzip",
                )
                ds_scatter.attrs["units"] = "counts"

                ds_continuum = spec_grp.create_dataset(
                    "compton_continuum",
                    data=np.asarray(spectrum_result.compton_continuum, dtype=np.float64),
                    compression="gzip",
                )
                ds_continuum.attrs["units"] = "counts"

                metrics_grp = spec_grp.create_group("metrics")
                metrics_grp.attrs["NX_class"] = "NXcollection"
                metrics_items = (
                    ("primary_fraction", spectrum_result.primary_fraction, "dimensionless"),
                    ("scatter_to_primary_ratio", spectrum_result.scatter_to_primary_ratio, "dimensionless"),
                    ("compton_to_primary_ratio", spectrum_result.compton_to_primary_ratio, "dimensionless"),
                    ("photopeak_efficiency", spectrum_result.photopeak_efficiency, "dimensionless"),
                    ("mean_scatter_order", spectrum_result.mean_scatter_order, "counts"),
                    ("window_purity", spectrum_result.window_purity, "dimensionless"),
                )
                for metric_name, metric_val, metric_unit in metrics_items:
                    ds_metric = metrics_grp.create_dataset(metric_name, data=float(metric_val))
                    ds_metric.attrs["units"] = metric_unit

                entry["data_spectrum"] = h5py.SoftLink("/entry/spectrum")

            # 4. Паспортные метаданные симуляции
            if meta_dict:
                sim_meta_grp = entry.create_group("simulation_metadata")
                sim_meta_grp.attrs["NX_class"] = "NXcollection"
                self._write_metadata_group(sim_meta_grp, meta_dict)

        return target_path
