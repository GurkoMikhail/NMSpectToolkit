"""Интерактивная 3D-визуализация геометрии гантри, детекторов и проекций ОФЭКТ на базе VTK."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import vtk


def load_vtm_blocks(vtm_path: str | Path) -> list[tuple[str, Path]]:
    """Разбирает XML-манифест `.vtm` (vtkMultiBlockDataSet) и возвращает список блоков `(имя, путь_к_vti)`.

    Args:
        vtm_path: Путь к файлу мультиблочного набора данных `.vtm`.

    Returns:
        Список кортежей `(block_name, vti_file_path)`.
    """
    resolved_vtm = Path(vtm_path)
    if not resolved_vtm.is_file():
        raise FileNotFoundError(f"Файл VTK MultiBlock (.vtm) не найден: {resolved_vtm}")

    tree = ET.parse(resolved_vtm)
    root = tree.getroot()
    parent_dir = resolved_vtm.parent

    blocks: list[tuple[str, Path]] = []
    for dataset_elem in root.iter("DataSet"):
        rel_file = dataset_elem.attrib.get("file", "")
        if not rel_file:
            continue
        block_name = dataset_elem.attrib.get("name", Path(rel_file).stem)
        vti_path = (parent_dir / rel_file).resolve()
        if vti_path.is_file():
            blocks.append((block_name, vti_path))

    if not blocks:
        raise ValueError(f"В файле {resolved_vtm} не найдено ни одного существующего блока DataSet (.vti).")
    return blocks


def build_vtk_scene_renderer(
    vtm_path: str | Path,
    background_color: tuple[float, float, float] = (0.10, 0.12, 0.16),
) -> vtk.vtkRenderer:
    """Создаёт и настраивает `vtkRenderer` со всеми 3D-блоками сцены ОФЭКТ из файла `.vtm`.

    Args:
        vtm_path: Путь к мультиблочному файлу `.vtm`.
        background_color: RGB-цвет фона сцены в диапазоне [0.0, 1.0].

    Returns:
        Настроенный экземпляр `vtk.vtkRenderer`.
    """
    blocks = load_vtm_blocks(vtm_path)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*background_color)

    scalar_min = 0.0
    scalar_max = 1.0
    projection_readers: list[tuple[str, vtk.vtkXMLImageDataReader]] = []
    structural_readers: list[tuple[str, vtk.vtkXMLImageDataReader]] = []

    for block_name, vti_path in blocks:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(str(vti_path))
        reader.Update()

        image_data = reader.GetOutput()
        point_data = image_data.GetPointData()
        cell_data = image_data.GetCellData()
        has_scalars = (
            (point_data is not None and point_data.GetScalars() is not None)
            or (cell_data is not None and cell_data.GetScalars() is not None)
        )

        lower_name = block_name.lower()
        is_structural = any(
            keyword in lower_name for keyword in ("phantom", "collimator", "casing", "glass", "shield", "body")
        )

        if has_scalars and not is_structural:
            val_range = image_data.GetScalarRange()
            if val_range[1] > scalar_max:
                scalar_max = float(val_range[1])
            projection_readers.append((block_name, reader))
        else:
            structural_readers.append((block_name, reader))

    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetNumberOfTableValues(256)
    lookup_table.SetTableRange(scalar_min, max(scalar_max, 1.0))
    lookup_table.SetHueRange(0.667, 0.0)
    lookup_table.Build()

    # Добавление проекционных плоскостей детекторов с цветовой картой счёта
    for _, proj_reader in projection_readers:
        geometry_filter = vtk.vtkImageDataGeometryFilter()
        geometry_filter.SetInputConnection(proj_reader.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(geometry_filter.GetOutputPort())
        mapper.SetLookupTable(lookup_table)
        mapper.SetScalarRange(scalar_min, max(scalar_max, 1.0))
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.92)
        renderer.AddActor(actor)

    # Добавление полупрозрачных каркасов конструктивных объёмов (фантом, коллиматоры, корпуса)
    for block_name, struct_reader in structural_readers:
        outline_filter = vtk.vtkOutlineFilter()
        outline_filter.SetInputConnection(struct_reader.GetOutputPort())

        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline_filter.GetOutputPort())

        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        lower_name = block_name.lower()
        if "phantom" in lower_name:
            outline_actor.GetProperty().SetColor(0.2, 0.85, 0.4)
            outline_actor.GetProperty().SetLineWidth(2.5)
        elif "collimator" in lower_name:
            outline_actor.GetProperty().SetColor(0.85, 0.75, 0.25)
            outline_actor.GetProperty().SetLineWidth(1.5)
        else:
            outline_actor.GetProperty().SetColor(0.65, 0.70, 0.75)
            outline_actor.GetProperty().SetLineWidth(1.0)
        renderer.AddActor(outline_actor)

    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lookup_table)
    scalar_bar.SetTitle("Счёт (имп.)")
    scalar_bar.SetNumberOfLabels(5)
    renderer.AddViewProp(scalar_bar)

    renderer.ResetCamera()
    return renderer


def launch_vtk_viewer(
    vtm_path: str | Path,
    window_size: Sequence[int] = (1280, 860),
    offscreen_screenshot_path: str | Path | None = None,
) -> None:
    """Открывает интерактивное 3D-окно VTK или сохраняет offscreen-рендер сцены в PNG.

    Args:
        vtm_path: Путь к мультиблочному файлу `.vtm`.
        window_size: Размеры окна `(width, height)` в пикселях.
        offscreen_screenshot_path: Если задан путь, выполняет headless-рендеринг в PNG без блокирующего окна.
    """
    renderer = build_vtk_scene_renderer(vtm_path)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(int(window_size[0]), int(window_size[1]))
    render_window.SetWindowName(f"NMSpectToolkit 3D VTK Viewer — {Path(vtm_path).name}")

    if offscreen_screenshot_path is not None:
        render_window.SetOffScreenRendering(1)
        render_window.Render()

        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(render_window)
        window_to_image.Update()

        out_png = Path(offscreen_screenshot_path)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        png_writer = vtk.vtkPNGWriter()
        png_writer.SetFileName(str(out_png))
        png_writer.SetInputConnection(window_to_image.GetOutputPort())
        png_writer.Write()
        return

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    axes_actor = vtk.vtkAxesActor()
    orientation_widget = vtk.vtkOrientationMarkerWidget()
    orientation_widget.SetOrientationMarker(axes_actor)
    orientation_widget.SetInteractor(interactor)
    orientation_widget.SetViewport(0.0, 0.0, 0.18, 0.18)
    orientation_widget.SetEnabled(1)
    orientation_widget.InteractiveOff()

    render_window.Render()
    interactor.Initialize()
    interactor.Start()
