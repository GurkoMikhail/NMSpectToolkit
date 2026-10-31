"""CLI-точки входа (`nmspect-view-vtk`, `nmspect-view-spectrum`) для запуска интерактивных визуализаторов."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from nmspect.visualization.spectrum_viewer import launch_spectrum_viewer
from nmspect.visualization.vtk_viewer import launch_vtk_viewer


def main_vtk(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI-команды `nmspect-view-vtk`."""
    parser = argparse.ArgumentParser(
        prog="nmspect-view-vtk",
        description="Интерактивный 3D-просмотрщик геометрии гантри, камер и проекций ОФЭКТ из файла .vtm.",
    )
    parser.add_argument(
        "vtm_path",
        type=str,
        help="Путь к файлу мультиблочной сборки VTK (.vtm).",
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        default=None,
        help="Сохранить offscreen-скриншот 3D-сцены в указанный PNG-файл и завершиться.",
    )
    args = parser.parse_args(argv)
    launch_vtk_viewer(args.vtm_path, offscreen_screenshot_path=args.screenshot)
    return 0


def main_spectrum(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI-команды `nmspect-view-spectrum`."""
    parser = argparse.ArgumentParser(
        prog="nmspect-view-spectrum",
        description="Интерактивный PyQt6 просмотрщик энергетических спектров из файлов .nxs или .npy.",
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        type=str,
        help="Один или несколько путей к файлам спектров (.nxs или .npy).",
    )
    parser.add_argument(
        "--auto-close",
        type=int,
        default=None,
        help="Автоматически закрыть окно через заданное время в миллисекундах.",
    )
    args = parser.parse_args(argv)
    return launch_spectrum_viewer(args.input_paths, auto_close_delay=args.auto_close)
