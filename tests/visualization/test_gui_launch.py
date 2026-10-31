"""Headless-тесты инициализации компонентов визуализации (PyQt6 SpectrumViewerWindow и VTK сцена)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hepunits import deg, keV, mm
from PyQt6 import QtWidgets

from nmspect.exporters import VtkParaviewExporter
from nmspect.visualization import (
    SpectrumViewerWindow,
    build_vtk_scene_renderer,
    load_vtm_blocks,
)


def test_pyqt6_spectrum_viewer_headless() -> None:
    """Проверяет создание и корректное закрытие окна SpectrumViewerWindow без утечки ресурсов."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    energies = np.linspace(0.0 * keV, 200.0 * keV, 128)
    counts = np.exp(-0.5 * ((energies - 74.6 * keV) / (3.5 * keV)) ** 2) * 500.0

    window = SpectrumViewerWindow(spectra=[("Тестовый спектр 161Tb", counts, energies)])
    assert window.windowTitle().startswith("NMSpectToolkit")
    window.close()
    window.deleteLater()


def test_vtk_scene_renderer_from_vtm(tmp_path: Path) -> None:
    """Проверяет разбор .vtm и сборку 3D-рендерера VTK в headless-режиме."""
    projections = np.ones((4, 12, 12), dtype=np.float64) * 25.0
    angles = np.array([0.0, 90.0, 180.0, 270.0]) * deg
    vtm_path = tmp_path / "scene.vtm"

    VtkParaviewExporter(vtm_path).export(
        projections=projections,
        angles=angles,
        pixel_size=4.2 * mm,
        detector_size=(120.0 * mm, 120.0 * mm, 9.5 * mm),
        orbit_radius=200.0 * mm,
    )

    blocks = load_vtm_blocks(vtm_path)
    assert len(blocks) >= 2

    renderer = build_vtk_scene_renderer(vtm_path)
    assert renderer.GetActors().GetNumberOfItems() > 0
