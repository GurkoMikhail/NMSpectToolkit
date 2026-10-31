"""
Загрузка входных проекционных данных, эталонных фантомов и карт ослабления,
а также экспорт реконструированных 2D/3D объёмов в форматы NumPy (`.npy`),
VTK ImageData (`.vti`) и графические превью ортогональных срезов (`.png`).

Все физические величины передаются и обрабатываются строго в системе единиц `hepunits`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import h5py
import matplotlib
import numpy as np
import pydicom
import vtk
from hepunits import cm, deg, mm
from numpy.typing import NDArray
from PIL import Image
from vtk.util.numpy_support import numpy_to_vtk

from nmspect.reconstruction.projector import normalize_projection_angles


def _extract_numeric_index(path: Path) -> tuple[int, str]:
    """Извлекает числовой индекс из имени файла `.dat` для корректной сортировки (`1.dat .. 120.dat`)."""
    matches = re.findall(r"(\d+)", path.stem)
    if matches:
        return int(matches[-1]), path.name
    return 10**9, path.name


def load_projections_from_dat(
    source_path: str | Path,
    angles: NDArray[np.float64] | Sequence[float] | None = None,
    pixel_size: float = 4.2 * mm,
    spr_size: tuple[int, int, int] = (128, 92, 120),
) -> tuple[NDArray[np.float32], NDArray[np.float64], float]:
    """
    Загружает стек проекций ОФЭКТ из директории с файлами `1.dat ... N.dat`
    либо из единого плоского файла `spr.dat`.

    :param source_path: Путь к папке с файлами `.dat` или к одиночному файлу `.dat`.
    :param angles: Опциональный массив углов в `hepunits`.
    :param pixel_size: Размер пикселя детектора в `hepunits`.
    :param spr_size: Размерность `(nx, ny, views)` при чтении плоского файла типа `spr.dat`.
    :return: Кортеж `(projections [views, nx, ny], angles [views], pixel_size)`.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Источник данных .dat не найден: {path}")

    if path.is_dir():
        dat_files = sorted(path.glob("*.dat"), key=_extract_numeric_index)
        if not dat_files:
            raise FileNotFoundError(f"В директории '{path}' не найдено файлов *.dat.")
        frames = [np.loadtxt(dat_file, dtype=np.float32) for dat_file in dat_files]
        projections = np.stack(frames, axis=0)
    else:
        raw = np.loadtxt(path, dtype=np.float32)
        if raw.ndim == 2:
            projections = raw[np.newaxis, :, :]
        else:
            nx, ny, num_views = spr_size
            if raw.size != nx * ny * num_views:
                raise ValueError(
                    f"Размер плоского файла '{path}' ({raw.size}) не соответствует "
                    f"ожидаемой форме spr_size={spr_size} ({nx * ny * num_views})."
                )
            projections = raw.reshape((num_views, nx, ny), order="C")

    num_views = int(projections.shape[0])
    norm_angles = normalize_projection_angles(angles, num_views)
    return projections, norm_angles, float(pixel_size)


def load_projections_from_nexus(
    file_path: str | Path,
) -> tuple[NDArray[np.float32], NDArray[np.float64], float]:
    """
    Загружает стек проекций, углы поворота гантри и размер пикселя из мастер-архива
    NeXus HDF5 (`.nxs` / `.h5`).

    :param file_path: Путь к файлу `.nxs` или `.h5`.
    :return: Кортеж `(projections [views, nx, ny], angles [views], pixel_size)`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл NeXus не найден: {path}")

    with h5py.File(path, "r") as h5_file:
        if "/entry/data/data" in h5_file:
            proj_ds = h5_file["/entry/data/data"]
        elif "/entry/data_projections/data" in h5_file:
            proj_ds = h5_file["/entry/data_projections/data"]
        elif "/entry/data_projections/projections" in h5_file:
            proj_ds = h5_file["/entry/data_projections/projections"]
        else:
            raise KeyError(
                "В файле NeXus не найден датасет проекций '/entry/data/data' или '/entry/data_projections/data'."
            )
        projections = np.asarray(proj_ds, dtype=np.float32)

        if "/entry/data/gantry_angle" in h5_file:
            raw_angles = np.asarray(h5_file["/entry/data/gantry_angle"], dtype=np.float64)
        elif "/entry/data_projections/gantry_angle" in h5_file:
            raw_angles = np.asarray(h5_file["/entry/data_projections/gantry_angle"], dtype=np.float64)
        elif "/entry/data/rotation_angle" in h5_file:
            raw_angles = np.asarray(h5_file["/entry/data/rotation_angle"], dtype=np.float64)
        else:
            raw_angles = None

        if "/entry/instrument/detector/pixel_size" in h5_file:
            pixel_size = float(np.asarray(h5_file["/entry/instrument/detector/pixel_size"]))
        else:
            pixel_size = 4.2 * mm

    norm_angles = normalize_projection_angles(raw_angles, int(projections.shape[0]))
    return projections, norm_angles, pixel_size


def load_projections_from_dicom(
    file_path: str | Path,
) -> tuple[NDArray[np.float32], NDArray[np.float64], float]:
    """
    Загружает стек проекций, углы и физический размер пикселя из клинического
    многокадрового файла DICOM Nuclear Medicine (`.dcm`), переводя ориентацию
    кадров обратно в систему координат проекта `(views, nx, ny)`.

    :param file_path: Путь к файлу `.dcm`.
    :return: Кортеж `(projections [views, nx, ny], angles [views], pixel_size)`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл DICOM не найден: {path}")

    dataset = pydicom.dcmread(str(path))
    pixel_array = np.asarray(dataset.pixel_array, dtype=np.float32)
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[np.newaxis, :, :]

    slope = float(dataset["RescaleSlope"].value) if "RescaleSlope" in dataset else 1.0
    intercept = (
        float(dataset["RescaleIntercept"].value) if "RescaleIntercept" in dataset else 0.0
    )
    proj_dicom = pixel_array * slope + intercept

    # Обратное преобразование из ориентации DICOM [views, rows (+Y..-Y), cols (+X..-X)]
    # к стандарту проекта [views, nx (-X..+X), ny (-Y..+Y)]
    projections = np.rot90(proj_dicom[:, ::-1, :], k=1, axes=(1, 2)).astype(np.float32)

    pixel_size = 4.2 * mm
    if "PixelSpacing" in dataset and len(dataset["PixelSpacing"].value) >= 1:
        pixel_size = float(dataset["PixelSpacing"].value[0]) * mm

    raw_angles: NDArray[np.float64] | None = None
    if "FramePrimaryAngleVector" in dataset:
        raw_angles = np.asarray(dataset["FramePrimaryAngleVector"].value, dtype=np.float64) * deg
    elif (
        "RotationInformationSequence" in dataset
        and len(dataset["RotationInformationSequence"].value) > 0
    ):
        rot_item = dataset["RotationInformationSequence"].value[0]
        if (0x0054, 0x0091) in rot_item:
            raw_angles = np.asarray(rot_item[(0x0054, 0x0091)].value, dtype=np.float64) * deg
        elif "StartAngle" in rot_item and "AngularStep" in rot_item:
            start_val = float(rot_item["StartAngle"].value) * deg
            step_val = float(rot_item["AngularStep"].value) * deg
            raw_angles = (
                start_val + np.arange(projections.shape[0], dtype=np.float64) * step_val
            )

    norm_angles = normalize_projection_angles(raw_angles, int(projections.shape[0]))
    return projections, norm_angles, pixel_size


def load_projections_from_npy(
    file_path: str | Path,
    angles_path: str | Path | None = None,
    pixel_size: float = 4.2 * mm,
) -> tuple[NDArray[np.float32], NDArray[np.float64], float]:
    """
    Загружает стек проекций из файла `.npy` и автоматически подгружает сопутствующий
    файл углов `*_angles.npy`, если он присутствует в той же директории.

    :param file_path: Путь к файлу проекций `.npy`.
    :param angles_path: Опциональный путь к файлу углов `.npy`.
    :param pixel_size: Физический размер пикселя в `hepunits`.
    :return: Кортеж `(projections, angles, pixel_size)`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл NumPy не найден: {path}")

    projections = np.asarray(np.load(path), dtype=np.float32)
    raw_angles: NDArray[np.float64] | None = None

    if angles_path is not None:
        ang_p = Path(angles_path)
        if ang_p.exists():
            raw_angles = np.asarray(np.load(ang_p), dtype=np.float64)
    else:
        stem_clean = re.sub(r"_MEW$", "", path.stem)
        candidates = [
            path.with_name(f"{path.stem}_angles.npy"),
            path.with_name(f"{stem_clean}_angles.npy"),
        ]
        for cand in candidates:
            if cand.exists():
                raw_angles = np.asarray(np.load(cand), dtype=np.float64)
                break

    norm_angles = normalize_projection_angles(raw_angles, int(projections.shape[0]))
    return projections, norm_angles, float(pixel_size)


def load_projections(
    source: str | Path | NDArray[np.float64],
    angles: NDArray[np.float64] | Sequence[float] | None = None,
    pixel_size: float = 4.2 * mm,
    spr_size: tuple[int, int, int] = (128, 92, 120),
) -> tuple[NDArray[np.float32], NDArray[np.float64], float]:
    """
    Универсальный загрузчик проекций ОФЭКТ: поддерживает готовые массивы `np.ndarray`,
    директории с файлами `1.dat ... N.dat`, файлы `.dat`, `.npy`, `.nxs`/`.h5` и `.dcm`.

    :param source: Массив проекций или путь к файлу/директории.
    :param angles: Опциональный вектор углов в `hepunits`.
    :param pixel_size: Размер пикселя по умолчанию в `hepunits`.
    :param spr_size: Размерность `(nx, ny, views)` при чтении плоского файла `.dat`.
    :return: Кортеж `(projections, angles, pixel_size)`.
    """
    if isinstance(source, np.ndarray):
        proj_arr = np.asarray(source, dtype=np.float32)
        norm_angles = normalize_projection_angles(angles, int(proj_arr.shape[0]))
        return proj_arr, norm_angles, float(pixel_size)

    path = Path(source)
    if path.is_dir() or path.suffix.lower() == ".dat":
        return load_projections_from_dat(
            path, angles=angles, pixel_size=pixel_size, spr_size=spr_size
        )
    if path.suffix.lower() in (".nxs", ".h5", ".hdf5"):
        proj, ang, px_size = load_projections_from_nexus(path)
        if angles is not None:
            ang = normalize_projection_angles(angles, int(proj.shape[0]))
        return proj, ang, px_size
    if path.suffix.lower() == ".dcm":
        proj, ang, px_size = load_projections_from_dicom(path)
        if angles is not None:
            ang = normalize_projection_angles(angles, int(proj.shape[0]))
        return proj, ang, px_size
    if path.suffix.lower() == ".npy":
        proj, ang, px_size = load_projections_from_npy(path, pixel_size=pixel_size)
        if angles is not None:
            ang = normalize_projection_angles(angles, int(proj.shape[0]))
        return proj, ang, px_size

    raise ValueError(f"Неподдерживаемый формат источника проекций: '{path}'")


def load_phantom_volume(
    source_path: str | Path,
    shape: tuple[int, int, int] = (128, 128, 92),
    order: Literal["A", "C", "F"] = "F",
) -> NDArray[np.float64]:
    """
    Загружает 3D функцию источника (эталонное распределение активности фантома)
    из файла `.dat` / `.npy` или директории фантома (например, `data/phantoms/nema`).

    :param source_path: Путь к файлу или папке фантома.
    :param shape: Ожидаемая 3D форма массива `(nx, ny_trans, nz)`.
    :param order: Порядок развёртки плоского массива (`'F'` для стандартных `.dat` фантомов проекта).
    :return: 3D массив распределения активности формы `shape`.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Источник фантома не найден: {path}")

    if path.is_dir():
        dat_candidates = sorted(
            dat_file
            for dat_file in path.glob("*.dat")
            if not dat_file.name.lower().startswith("a")
        )
        if not dat_candidates:
            dat_candidates = sorted(path.glob("*.dat"))
        if not dat_candidates:
            raise FileNotFoundError(f"В директории фантома '{path}' не найдено файлов *.dat.")
        path = dat_candidates[0]

    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float64)
        return np.asarray(arr.reshape(shape, order=order) if arr.ndim == 1 else arr, dtype=np.float64)

    raw = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64)
    return np.asarray(raw.reshape(shape, order=order), dtype=np.float64)


def load_attenuation_volume(
    source_path: str | Path,
    shape: tuple[int, int, int] = (128, 128, 92),
    order: Literal["A", "C", "F"] = "F",
) -> NDArray[np.float64]:
    """
    Загружает 2D или 3D карту линейных коэффициентов ослабления из файла `.dat` / `.npy`
    или директории фантома и приводит значения к стандарту `hepunits`.

    Если численные коэффициенты ослабления в файле заданы в традиционных обратных
    сантиметрах (например, `0.15` 1/см для воды и `0.04` 1/см для лёгочной вставки),
    выполняется автоматический перевод в `hepunits` делением на константу `cm`.

    :param source_path: Путь к файлу карты ослабления или директории фантома.
    :param shape: Ожидаемая 3D форма массива при чтении плоского файла `.dat`.
    :param order: Порядок развёртки плоского файла `.dat` (`'F'` по умолчанию).
    :return: Массив линейных коэффициентов ослабления в единицах `hepunits`.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Карта ослабления не найдена: {path}")

    if path.is_dir():
        att_candidates = sorted(
            dat_file
            for dat_file in path.glob("*.dat")
            if dat_file.name.lower().startswith("a")
        )
        if not att_candidates:
            att_candidates = sorted(path.glob("*.dat"))
        if not att_candidates:
            raise FileNotFoundError(
                f"В директории '{path}' не найдено файлов карты ослабления *.dat."
            )
        path = att_candidates[0]

    if path.suffix.lower() == ".npy":
        arr = np.asarray(np.load(path), dtype=np.float64)
        if arr.ndim == 1:
            arr = np.asarray(arr.reshape(shape, order=order), dtype=np.float64)
    else:
        raw = np.asarray(np.loadtxt(path, dtype=np.float64), dtype=np.float64)
        arr = np.asarray(raw.reshape(shape, order=order) if raw.ndim == 1 else raw, dtype=np.float64)

    # В hepunits 1/cm == 0.1, поэтому для биологических тканей (вода ~0.15/cm, кость ~0.30/cm)
    # значения в hepunits лежат в диапазоне ~0.004..0.04. Если максимум > 0.06, данные в 1/см.
    max_val = float(np.max(arr)) if arr.size > 0 else 0.0
    if max_val > 0.06:
        arr = arr / cm

    return arr


def save_volume_as_npy(
    volume: NDArray[np.float64],
    output_path: str | Path,
) -> Path:
    """
    Сохраняет реконструированный объём в бинарный файл NumPy (`.npy`).

    :param volume: Реконструированный 2D срез или 3D объём.
    :param output_path: Целевой путь сохранения.
    :return: Путь к сохранённому файлу.
    """
    out = Path(output_path)
    if out.suffix.lower() != ".npy":
        out = out.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(volume, dtype=np.float32))
    return out


def save_volume_as_vti(
    volume: NDArray[np.float64],
    output_path: str | Path,
    voxel_size: float = 4.2 * mm,
    attenuation_map: NDArray[np.float64] | None = None,
) -> Path:
    """
    Сохраняет реконструированный 3D/2D объём в регулярную сетку VTK ImageData (`.vti`)
    для интерактивной визуализации в ParaView в единой системе координат проекта.

    :param volume: Реконструированный объём `(nx, nx, ny)` или 2D срез `(nx, nx)`.
    :param output_path: Путь к выходному файлу `.vti`.
    :param voxel_size: Физический размер вокселя в `hepunits`.
    :param attenuation_map: Опциональная карта затухания для включения вторым скалярным полем.
    :return: Путь к созданному файлу `.vti`.
    """
    out = Path(output_path)
    if out.suffix.lower() != ".vti":
        out = out.with_suffix(".vti")
    out.parent.mkdir(parents=True, exist_ok=True)

    vol_arr = np.asarray(volume, dtype=np.float32)
    if vol_arr.ndim == 2:
        vol_arr = vol_arr[:, :, np.newaxis]

    nx, ny_trans, nz = vol_arr.shape
    step = float(voxel_size)

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx + 1, ny_trans + 1, nz + 1)
    image_data.SetSpacing(step, step, step)
    image_data.SetOrigin(-0.5 * nx * step, -0.5 * ny_trans * step, -0.5 * nz * step)

    vtk_activity = numpy_to_vtk(np.ascontiguousarray(vol_arr.ravel(order="F")), deep=True)
    vtk_activity.SetName("reconstructed_activity")
    image_data.GetCellData().SetScalars(vtk_activity)

    if attenuation_map is not None:
        att_arr = np.asarray(attenuation_map, dtype=np.float32)
        if att_arr.ndim == 2:
            att_arr = np.repeat(att_arr[:, :, np.newaxis], nz, axis=2)
        if att_arr.shape == vol_arr.shape:
            vtk_att = numpy_to_vtk(np.ascontiguousarray(att_arr.ravel(order="F")), deep=True)
            vtk_att.SetName("attenuation_map")
            image_data.GetCellData().AddArray(vtk_att)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(out))
    writer.SetInputData(image_data)
    writer.Write()
    return out


def _apply_colormap_uint8(
    gray_uint8: NDArray[np.uint8],
    cmap_name: str | None = "viridis",
) -> NDArray[np.uint8]:
    """
    Преобразует 2D массив яркости `uint8 (0..255)` в RGB массив `uint8` формы `(H, W, 3)`
    с использованием указанной цветовой карты (по умолчанию `'viridis'`).
    При `cmap_name is None` или `'gray'`/`'grayscale'` возвращает исходный 2D массив.
    """
    if cmap_name is None or cmap_name.strip().lower() in ("gray", "grayscale", "l", "none"):
        return gray_uint8

    cmap_obj = matplotlib.colormaps[cmap_name.strip()]
    lut = np.round(cmap_obj(np.linspace(0.0, 1.0, 256))[:, :3] * 255.0).astype(np.uint8)
    return lut[gray_uint8]


def save_slices_preview_png(
    volume: NDArray[np.float64],
    output_path: str | Path,
    cmap: str | None = "viridis",
    transaxial_slice_index: int | None = None,
) -> Path:
    """
    Генерирует и сохраняет изображение (`.png`) с ортогональными срезами
    (трансаксиальным, корональным и сагиттальным) для 3D объёма либо одиночным
    срезом для 2D реконструкции в заданной цветовой карте (по умолчанию `'viridis'`).

    :param volume: Реконструированный 2D срез `(nx, nx)` или 3D объём `(nx, nx, ny)`.
    :param output_path: Путь к создаваемому PNG-файлу.
    :param cmap: Имя цветовой палитры (`'viridis'` по умолчанию, `None`/`'gray'` для ч/б).
    :param transaxial_slice_index: Индекс аксиального среза Z (по умолчанию центральный `nz // 2`).
    :return: Путь к сохранённому файлу `.png`.
    """
    out = Path(output_path)
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)

    vol_arr = np.clip(np.asarray(volume, dtype=np.float64), 0.0, None)

    def _norm_uint8(slice_2d: NDArray[np.float64]) -> NDArray[np.uint8]:
        slice_max = float(np.max(slice_2d))
        if slice_max <= 0.0:
            return np.zeros(slice_2d.shape, dtype=np.uint8)
        return np.round((slice_2d / slice_max) * 255.0).astype(np.uint8)

    if vol_arr.ndim == 2:
        canvas_gray = _norm_uint8(vol_arr.T[::-1, :])
    else:
        nx, ny_trans, nz = vol_arr.shape
        z_idx = (
            int(np.clip(transaxial_slice_index, 0, nz - 1))
            if transaxial_slice_index is not None
            else (nz // 2)
        )
        transaxial = _norm_uint8(vol_arr[:, :, z_idx].T[::-1, :])
        coronal = _norm_uint8(vol_arr[:, ny_trans // 2, :].T[::-1, :])
        sagittal = _norm_uint8(vol_arr[nx // 2, :, :].T[::-1, :])

        # Выравниваем высоту панелей до максимальной для аккуратной горизонтальной склейки
        target_height = max(transaxial.shape[0], coronal.shape[0], sagittal.shape[0])
        panels = []
        for panel in (transaxial, coronal, sagittal):
            pad_top = (target_height - panel.shape[0]) // 2
            pad_bottom = target_height - panel.shape[0] - pad_top
            padded_panel = np.pad(
                panel, ((pad_top, pad_bottom), (4, 4)), mode="constant", constant_values=0
            )
            panels.append(padded_panel)
        canvas_gray = np.concatenate(panels, axis=1)

    colored = _apply_colormap_uint8(canvas_gray, cmap_name=cmap)
    if colored.ndim == 2:
        Image.fromarray(colored, mode="L").save(out)
    else:
        Image.fromarray(colored, mode="RGB").save(out)
    return out
