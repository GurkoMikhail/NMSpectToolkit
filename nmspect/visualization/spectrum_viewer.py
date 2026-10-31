"""Интерактивный графический просмотрщик энергетических спектров ОФЭКТ на базе PyQt6 и pyqtgraph."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pyqtgraph as pg
from hepunits import keV
from numpy.typing import NDArray
from PyQt6 import QtCore, QtWidgets


class SpectrumViewerWindow(QtWidgets.QMainWindow):
    """Главное окно визуализации энергетических спектров сцинтилляционного детектора."""

    def __init__(
        self,
        spectra: Sequence[tuple[str, NDArray[np.floating], NDArray[np.floating] | None]],
        window_title: str = "NMSpectToolkit — Анализатор энергетических спектров",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Инициализирует окно с вкладками или кривыми энергетических спектров.

        Args:
            spectra: Последовательность кортежей (название, вектор_счёта, вектор_энергий_в_hepunits_или_None).
            window_title: Заголовок главного окна.
            parent: Родительский виджет Qt.
        """
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.resize(1024, 680)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        self.plot_widget = pg.PlotWidget(title="Энергетический спектр зарегистрированных вспышек")
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("bottom", "Энергия", units="кэВ")
        self.plot_widget.setLabel("left", "Число отсчётов (имп.)")
        self.plot_widget.addLegend()

        main_layout.addWidget(self.plot_widget)

        palette = [
            (31, 119, 180),
            (214, 39, 40),
            (44, 160, 44),
            (255, 127, 14),
            (148, 103, 189),
        ]

        for curve_idx, (label, counts_arr, energy_bins) in enumerate(spectra):
            counts = np.asarray(counts_arr, dtype=np.float64).ravel()
            if energy_bins is not None:
                axis_x = np.asarray(energy_bins, dtype=np.float64).ravel() / keV
            else:
                axis_x = np.arange(counts.size, dtype=np.float64)

            color = palette[curve_idx % len(palette)]
            pen = pg.mkPen(color=color, width=2)
            self.plot_widget.plot(axis_x, counts, pen=pen, name=label)


def load_spectra_from_path(
    input_path: str | Path,
) -> list[tuple[str, NDArray[np.float64], NDArray[np.float64] | None]]:
    """Загружает энергетические спектры из файла NeXus (.nxs) или массива NumPy (.npy).

    Args:
        input_path: Путь к файлу `.nxs` или `.npy`.

    Returns:
        Список кортежей (имя_компоненты, массив_счёта, массив_энергий_в_hepunits_или_None).
    """
    resolved_path = Path(input_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Файл спектра не найден: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix in (".nxs", ".h5", ".hdf", ".hdf5"):
        curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64] | None]] = []
        with h5py.File(resolved_path, "r") as h5_file:
            if "entry/spectrum" not in h5_file:
                raise KeyError(f"Группа 'entry/spectrum' отсутствует в файле NeXus: {resolved_path}")
            spec_group = h5_file["entry/spectrum"]
            energy_bins: NDArray[np.float64] | None = None
            if "energy" in spec_group:
                raw_energy = np.asarray(spec_group["energy"][:], dtype=np.float64)
                units_attr = spec_group["energy"].attrs.get("units", "keV")
                if isinstance(units_attr, bytes):
                    units_attr = units_attr.decode("utf-8", errors="ignore")
                if str(units_attr).lower() == "kev":
                    energy_bins = raw_energy * keV
                else:
                    energy_bins = raw_energy

            component_mapping = [
                ("total", "Полный спектр (Total)"),
                ("primary", "Первичные фотоны (Primary)"),
                ("scatter", "Рассеянные фотоны (Scatter)"),
                ("compton_continuum", "Комптоновский фон (Continuum)"),
            ]
            for dataset_key, display_label in component_mapping:
                if dataset_key in spec_group:
                    counts_data = np.asarray(spec_group[dataset_key][:], dtype=np.float64)
                    curves.append((display_label, counts_data, energy_bins))
        if not curves:
            raise ValueError(f"В группе 'entry/spectrum' файла {resolved_path} не найдено массивов спектра.")
        return curves

    if suffix == ".npy":
        raw_array = np.asarray(np.load(resolved_path), dtype=np.float64)
        if raw_array.ndim == 1:
            return [(resolved_path.stem, raw_array, None)]
        if raw_array.ndim == 2:
            return [
                (f"{resolved_path.stem} (проекция {row_idx})", raw_array[row_idx], None)
                for row_idx in range(min(raw_array.shape[0], 5))
            ]
        raise ValueError(f"Неподдерживаемая размерность массива спектра в {resolved_path}: {raw_array.shape}")

    raise ValueError(f"Неподдерживаемый формат файла спектра '{suffix}': {resolved_path}")


def launch_spectrum_viewer(
    input_paths: Sequence[str | Path],
    auto_close_delay: int | None = None,
) -> int:
    """Запускает графическое приложение просмотра энергетических спектров.

    Args:
        input_paths: Список путей к файлам `.nxs` или `.npy` со спектрами.
        auto_close_delay: Опциональная задержка автозакрытия окна (в мс) для автоматизированных проверок.

    Returns:
        Код завершения цикла событий Qt.
    """
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        created_app = True

    all_spectra: list[tuple[str, NDArray[np.float64], NDArray[np.float64] | None]] = []
    for path_item in input_paths:
        loaded = load_spectra_from_path(path_item)
        prefix = Path(path_item).stem
        for label, counts, energies in loaded:
            all_spectra.append((f"{prefix}: {label}", counts, energies))

    viewer = SpectrumViewerWindow(spectra=all_spectra)
    viewer.show()

    if auto_close_delay is not None and auto_close_delay > 0:
        QtCore.QTimer.singleShot(auto_close_delay, viewer.close)

    if created_app and isinstance(app, QtWidgets.QApplication):
        return int(app.exec())
    return 0
