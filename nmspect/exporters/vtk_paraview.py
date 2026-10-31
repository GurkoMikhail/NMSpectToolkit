"""Экспорт 3D-геометрии гантри, детекторов, фантома и проекций ОФЭКТ в формат VTK ParaView (.vtm + .vti)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import vtk
from hepunits import deg, keV, mm
from numpy.typing import NDArray
from vtk.util import numpy_support

from nmspect.spectrometry.models import EnergyWindowConfig


class VtkParaviewExporter:
    """Экспортирует геометрию ОФЭКТ-сканера, фантома и 2D/3D проекции в мультиблочный набор `.vtm` + `.vti`."""

    def __init__(self, output_path: str | Path | None = None) -> None:
        """Инициализирует экспортер VTK ParaView.

        Args:
            output_path: Опциональный путь к целевому файлу `.vtm` (может быть также передан в `export`).
        """
        self.output_path: Path | None = Path(output_path) if output_path is not None else None

    @staticmethod
    def _write_structural_box_vti(
        file_path: Path,
        center: tuple[float, float, float],
        lengths: tuple[float, float, float],
    ) -> None:
        """Создаёт и сохраняет структурный 3D-бокс (`vtkImageData` без скаляров) в файл `.vti`."""
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(2, 2, 2)
        image_data.SetSpacing(float(lengths[0]), float(lengths[1]), float(lengths[2]))
        image_data.SetOrigin(
            float(center[0] - 0.5 * lengths[0]),
            float(center[1] - 0.5 * lengths[1]),
            float(center[2] - 0.5 * lengths[2]),
        )

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(image_data)
        writer.Write()

    @staticmethod
    def _write_projection_plane_vti(
        file_path: Path,
        projection_2d: NDArray[np.floating],
        pixel_size: float,
        center_y: float,
        thickness_y: float,
        window_config: EnergyWindowConfig | None = None,
    ) -> None:
        """Сохраняет 2D-проекцию на плоскости кристалла детектора в мировых координатах в файл `.vti`."""
        proj_arr = np.asarray(projection_2d, dtype=np.float32)
        num_x, num_z = proj_arr.shape
        step_val = float(pixel_size / mm)
        thick_val = max(float(thickness_y / mm), 1.0)
        pos_y = float(center_y / mm)

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(num_x + 1, 2, num_z + 1)
        image_data.SetSpacing(step_val, thick_val, step_val)
        image_data.SetOrigin(
            -0.5 * float(num_x) * step_val,
            pos_y - 0.5 * thick_val,
            -0.5 * float(num_z) * step_val,
        )

        flat_scalars = np.ascontiguousarray(proj_arr.ravel(order="F"), dtype=np.float32)
        vtk_scalars = numpy_support.numpy_to_vtk(flat_scalars, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_scalars.SetName("counts")
        image_data.GetCellData().SetScalars(vtk_scalars)

        if window_config is not None and window_config.primary_peak is not None:
            peak_arr = vtk.vtkFloatArray()
            peak_arr.SetName("primary_peak")
            peak_arr.InsertNextValue(float(window_config.primary_peak / keV))
            image_data.GetFieldData().AddArray(peak_arr)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(image_data)
        writer.Write()

    @staticmethod
    def _write_projections_stack_vti(
        file_path: Path,
        projections_3d: NDArray[np.floating],
        pixel_size: float,
        window_config: EnergyWindowConfig | None = None,
    ) -> None:
        """Сохраняет полный 3D-стек проекций `(views, nx, ny)` в регулярную сетку `.vti`."""
        proj_arr = np.asarray(projections_3d, dtype=np.float32)
        num_views, num_x, num_y = proj_arr.shape
        step_val = float(pixel_size / mm)

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(num_x + 1, num_y + 1, num_views + 1)
        image_data.SetSpacing(step_val, step_val, 1.0)
        image_data.SetOrigin(-0.5 * float(num_x) * step_val, -0.5 * float(num_y) * step_val, 0.0)

        transposed = np.ascontiguousarray(proj_arr.transpose(1, 2, 0).ravel(order="F"), dtype=np.float32)
        vtk_scalars = numpy_support.numpy_to_vtk(transposed, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_scalars.SetName("projections")
        image_data.GetCellData().SetScalars(vtk_scalars)

        if window_config is not None and window_config.primary_peak is not None:
            peak_arr = vtk.vtkFloatArray()
            peak_arr.SetName("primary_peak")
            peak_arr.InsertNextValue(float(window_config.primary_peak / keV))
            image_data.GetFieldData().AddArray(peak_arr)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(str(file_path))
        writer.SetInputData(image_data)
        writer.Write()

    def export(
        self,
        projections: NDArray[np.floating],
        angles: Sequence[float] | NDArray[np.floating] | None = None,
        pixel_size: float = 4.2 * mm,
        detector_size: Sequence[float] | None = None,
        orbit_radius: float = 250.0 * mm,
        window_config: EnergyWindowConfig | None = None,
        metadata: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Формирует и сохраняет мультиблочную 3D-сцену VTK ParaView (`.vtm` + `.vti`).

        Args:
            projections: 3D стек проекций `(views, nx, ny)` или 2D срез `(nx, ny)`.
            angles: Опциональный вектор углов проекций в базовых единицах hepunits (радианах).
            pixel_size: Размер пикселя проекции в базовых единицах hepunits (mm).
            detector_size: Физические габариты кристалла детектора `(size_x, size_y, size_z)` в hepunits (mm).
            orbit_radius: Радиус орбиты вращения детекторов в базовых единицах hepunits (mm).
            window_config: Конфигурация энергетического окна `EnergyWindowConfig`.
            metadata: Опциональные метаданные геометрии сканера и фантома.
            output_path: Целевой путь сохранения файла `.vtm` (переопределяет путь из конструктора).

        Returns:
            Полный путь к созданному файлу `.vtm`.
        """
        target_path = Path(output_path) if output_path is not None else self.output_path
        if target_path is None:
            raise ValueError("Не указан путь сохранения файла VTK MultiBlock (output_path).")
        if pixel_size <= 0.0:
            raise ValueError(f"Размер пикселя должен быть строго положительным: {pixel_size}")
        if orbit_radius <= 0.0:
            raise ValueError(f"Радиус орбиты должен быть строго положительным: {orbit_radius}")

        proj_arr = np.asarray(projections, dtype=np.float32)
        if proj_arr.ndim == 2:
            proj_arr = proj_arr[np.newaxis, ...]
        elif proj_arr.ndim != 3 or proj_arr.size == 0:
            raise ValueError(f"Ожидался непустой 2D или 3D массив проекций, получена форма: {proj_arr.shape}")

        angles_arr: NDArray[np.float64] | None = None
        if angles is not None:
            angles_arr = np.asarray(angles, dtype=np.float64).ravel()
            if angles_arr.size != proj_arr.shape[0]:
                raise ValueError(
                    f"Число углов ({angles_arr.size}) не совпадает с числом проекций ({proj_arr.shape[0]})."
                )

        if target_path.suffix.lower() != ".vtm":
            target_path = target_path.with_suffix(".vtm")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        blocks_dir = target_path.parent / target_path.stem
        blocks_dir.mkdir(parents=True, exist_ok=True)

        det_dims = (
            (float(detector_size[0] / mm), float(detector_size[1] / mm), float(detector_size[2] / mm))
            if detector_size is not None and len(detector_size) >= 3
            else (
                float(proj_arr.shape[1]) * float(pixel_size / mm),
                float(proj_arr.shape[2]) * float(pixel_size / mm),
                9.5,
            )
        )
        radius_val = float(orbit_radius / mm)
        crystal_thick = det_dims[2]
        collimator_thick = 35.0
        casing_thick = 60.0
        collimator_center_y = radius_val - 0.5 * crystal_thick - 0.5 * collimator_thick
        casing_center_y = radius_val + 0.5 * crystal_thick + 0.5 * casing_thick

        block_entries: list[tuple[str, str]] = []

        # 1. Структурные 3D-объёмы сцены (Phantom, Collimator_1/2, Casing_1/2)
        phantom_vti = blocks_dir / "Phantom.vti"
        self._write_structural_box_vti(
            file_path=phantom_vti,
            center=(0.0, 0.0, 0.0),
            lengths=(300.0, 220.0, 220.0),
        )
        block_entries.append(("Phantom", f"{blocks_dir.name}/{phantom_vti.name}"))

        collimator_1_vti = blocks_dir / "Collimator_1.vti"
        self._write_structural_box_vti(
            file_path=collimator_1_vti,
            center=(0.0, collimator_center_y, 0.0),
            lengths=(det_dims[0], collimator_thick, det_dims[1]),
        )
        block_entries.append(("Collimator_1", f"{blocks_dir.name}/{collimator_1_vti.name}"))

        collimator_2_vti = blocks_dir / "Collimator_2.vti"
        self._write_structural_box_vti(
            file_path=collimator_2_vti,
            center=(0.0, -collimator_center_y, 0.0),
            lengths=(det_dims[0], collimator_thick, det_dims[1]),
        )
        block_entries.append(("Collimator_2", f"{blocks_dir.name}/{collimator_2_vti.name}"))

        casing_1_vti = blocks_dir / "Casing_1.vti"
        self._write_structural_box_vti(
            file_path=casing_1_vti,
            center=(0.0, casing_center_y, 0.0),
            lengths=(det_dims[0] + 20.0, casing_thick, det_dims[1] + 20.0),
        )
        block_entries.append(("Casing_1", f"{blocks_dir.name}/{casing_1_vti.name}"))

        casing_2_vti = blocks_dir / "Casing_2.vti"
        self._write_structural_box_vti(
            file_path=casing_2_vti,
            center=(0.0, -casing_center_y, 0.0),
            lengths=(det_dims[0] + 20.0, casing_thick, det_dims[1] + 20.0),
        )
        block_entries.append(("Casing_2", f"{blocks_dir.name}/{casing_2_vti.name}"))

        # 2. Текстурированные проекции на кристаллах детекторов 1 (+Y при 0°) и 2 (-Y при 180°)
        det1_proj_vti = blocks_dir / "Detector_1_Projection.vti"
        self._write_projection_plane_vti(
            file_path=det1_proj_vti,
            projection_2d=proj_arr[0],
            pixel_size=pixel_size,
            center_y=orbit_radius,
            thickness_y=crystal_thick * mm,
            window_config=window_config,
        )
        block_entries.append(("Detector_1_Projection", f"{blocks_dir.name}/{det1_proj_vti.name}"))

        if angles_arr is not None and proj_arr.shape[0] > 1:
            full_turn = 360.0 * deg
            half_turn = 180.0 * deg
            target_opposite_angle = (float(angles_arr[0]) + half_turn) % full_turn
            angular_diffs = np.abs(
                (angles_arr - target_opposite_angle + half_turn) % full_turn - half_turn
            )
            second_view_idx = int(np.argmin(angular_diffs))
        else:
            second_view_idx = proj_arr.shape[0] // 2

        det2_proj_vti = blocks_dir / "Detector_2_Projection.vti"
        self._write_projection_plane_vti(
            file_path=det2_proj_vti,
            projection_2d=proj_arr[second_view_idx][::-1, :],
            pixel_size=pixel_size,
            center_y=-orbit_radius,
            thickness_y=crystal_thick * mm,
            window_config=window_config,
        )
        block_entries.append(("Detector_2_Projection", f"{blocks_dir.name}/{det2_proj_vti.name}"))

        # 3. Полный 3D-стек проекций
        stack_vti = blocks_dir / "Projections_Volume.vti"
        self._write_projections_stack_vti(
            file_path=stack_vti,
            projections_3d=proj_arr,
            pixel_size=pixel_size,
            window_config=window_config,
        )
        block_entries.append(("Projections_Volume", f"{blocks_dir.name}/{stack_vti.name}"))

        # 4. Формирование XML-манифеста .vtm
        vtk_file_elem = ET.Element(
            "VTKFile",
            attrib={
                "type": "vtkMultiBlockDataSet",
                "version": "1.0",
                "byte_order": "LittleEndian",
                "header_type": "UInt32",
            },
        )
        multiblock_elem = ET.SubElement(vtk_file_elem, "vtkMultiBlockDataSet")
        for block_idx, (block_name, rel_file) in enumerate(block_entries):
            ET.SubElement(
                multiblock_elem,
                "DataSet",
                attrib={
                    "index": str(block_idx),
                    "name": block_name,
                    "file": rel_file,
                },
            )

        xml_tree = ET.ElementTree(vtk_file_elem)
        ET.indent(xml_tree, space="  ")
        xml_tree.write(target_path, encoding="utf-8", xml_declaration=True)
        return target_path
